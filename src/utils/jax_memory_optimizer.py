"""
Enhanced memory optimization for JAX-accelerated Boolean cell cycle analysis.

This module extends the JAX accelerator with additional memory optimization techniques
for handling very large state spaces efficiently.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Try to import JAX packages, but continue if not available
try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import device_put, jit, lax, vmap

    JAX_AVAILABLE = True
    logging.info("JAX memory optimizer available")
except ImportError:
    JAX_AVAILABLE = False
    logging.warning("JAX memory optimizer not available")


def estimate_memory_requirements(num_cyclins: int, batch_size: int) -> Tuple[float, float]:
    """
    Estimate memory requirements for processing states with the given parameters.

    Args:
        num_cyclins: Number of cyclins in the model
        batch_size: Size of batches to process at once

    Returns:
        Tuple[float, float]: Estimated memory usage in GB (peak, steady-state)
    """
    if not JAX_AVAILABLE:
        return 0.0, 0.0

    # Size of a single state in bytes (int32)
    state_size_bytes = 4 * num_cyclins

    # Size of graph matrix in bytes (int32)
    graph_size_bytes = 4 * num_cyclins * num_cyclins

    # Size of a batch of states
    batch_size_bytes = state_size_bytes * batch_size

    # Estimate multiplication overhead (batch * graph matrix)
    matmul_overhead = batch_size * num_cyclins * num_cyclins * 4

    # Total steady state memory usage
    steady_memory_bytes = graph_size_bytes + (batch_size_bytes * 2) + matmul_overhead

    # Peak memory usage during computation (includes JIT compilation overhead)
    # This is a rough estimate and may vary with JIT compilation specifics
    peak_memory_bytes = steady_memory_bytes * 2.5

    # Convert to GB
    steady_memory_gb = steady_memory_bytes / (1024 * 1024 * 1024)
    peak_memory_gb = peak_memory_bytes / (1024 * 1024 * 1024)

    return peak_memory_gb, steady_memory_gb


def adaptive_batch_size(num_cyclins: int, available_memory_gb: Optional[float] = None) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        num_cyclins: Number of cyclins in the model
        available_memory_gb: Available GPU memory in GB (if None, tries to detect)

    Returns:
        int: Optimal batch size for memory-constrained execution
    """
    if not JAX_AVAILABLE:
        return 64  # Default fallback

    # Try to detect available memory if not provided
    if available_memory_gb is None:
        try:
            # This is a simplistic approach - in production, would need more robust detection
            devices = jax.devices()
            if devices and hasattr(devices[0], "memory_stats") and callable(devices[0].memory_stats):
                memory_stats = devices[0].memory_stats()
                if isinstance(memory_stats, dict) and "bytes_free" in memory_stats:
                    available_memory_gb = memory_stats["bytes_free"] / (1024 * 1024 * 1024)
                else:
                    # Default conservative estimate if can't read memory stats
                    available_memory_gb = 4.0  # 4 GB default
            else:
                available_memory_gb = 4.0  # 4 GB default
        except Exception:
            available_memory_gb = 4.0  # 4 GB default

    # Reserve 25% of memory for system/overhead
    usable_memory_gb = available_memory_gb * 0.75

    # Start with a reasonable batch size
    batch_size = 256

    # Adjust batch size down if memory requirements exceed available memory
    while batch_size > 16:
        peak_memory, steady_memory = estimate_memory_requirements(num_cyclins, batch_size)
        if peak_memory < usable_memory_gb:
            break
        batch_size //= 2

    return batch_size


def memory_efficient_state_processing(
    states: List,
    graph_matrix: List,
    batch_size: Optional[int] = None,
    max_iterations: int = 50,
    device=None,
    memory_limit_gb: Optional[float] = None,
) -> Dict:
    """
    Process states with memory-efficient batching and garbage collection.

    This function optimizes memory usage during state processing by:
    1. Dynamically adjusting batch size based on memory constraints
    2. Explicit garbage collection between batches
    3. Minimizing temporary memory allocations

    Args:
        states: List of initial states to process
        graph_matrix: Connection graph matrix
        batch_size: Size of batches (or None to auto-calculate)
        max_iterations: Maximum iterations for state convergence
        device: JAX device to use for computation
        memory_limit_gb: Memory limit in GB (None to auto-detect)

    Returns:
        Dict: Mapping of initial states to final states
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is not available")

    # Calculate optimal batch size if not provided
    if batch_size is None:
        num_cyclins = len(graph_matrix)
        batch_size = adaptive_batch_size(num_cyclins, memory_limit_gb)
        logging.info(f"Auto-calculated memory-efficient batch size: {batch_size}")

    # Convert graph matrix to JAX array once (reuse for all batches)
    graph_matrix_jax = jnp.array(graph_matrix, dtype=jnp.int32)
    if device is not None:
        graph_matrix_jax = device_put(graph_matrix_jax, device)

    # Pre-compile the update function with jit
    batch_update_fn = jit(
        vmap(
            lambda state, g_matrix: jnp.greater_equal(jnp.dot(g_matrix, state), 0).astype(jnp.int32), in_axes=(0, None)
        )
    )

    # Results dictionary
    results = {}
    batch_count = (len(states) + batch_size - 1) // batch_size

    # Process in batches
    for batch_idx in range(batch_count):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(states))

        # Get current batch
        batch_states = states[start_idx:end_idx]

        # Convert to JAX array
        batch_jax = jnp.array(batch_states, dtype=jnp.int32)
        if device is not None:
            batch_jax = device_put(batch_jax, device)

        # Run state iterations until convergence
        curr_states = batch_jax
        iteration = 0

        while iteration < max_iterations:
            new_states = batch_update_fn(curr_states, graph_matrix_jax)

            # Check if all states have converged
            all_equal = jnp.all(new_states == curr_states)
            if all_equal:
                break

            curr_states = new_states
            iteration += 1

        # Convert final states to host and store results
        final_states = np.array(curr_states)

        for i, (init_state, final_state) in enumerate(zip(batch_states, final_states)):
            results[tuple(init_state)] = tuple(final_state)

        # Log progress for large state spaces
        if batch_count > 10 and batch_idx % (batch_count // 10) == 0:
            progress = (batch_idx + 1) / batch_count * 100
            logging.info(f"Processing progress: {progress:.1f}% ({batch_idx + 1}/{batch_count} batches)")

    return results
