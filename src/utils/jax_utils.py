"""
JAX utility functions for accelerating Boolean Cell Cycle state calculations.

This module provides JAX-based optimizations for the state calculations,
allowing for GPU acceleration and batch processing.
"""

import jax
import jax.numpy as jnp
from jax import device_put, jit, vmap


@jit
def jax_update_single_state(state, graph_matrix):
    """
    JAX-optimized version of state update logic for a single state.

    Args:
        state: Current state array (jnp.array)
        graph_matrix: Connection graph matrix (jnp.array)

    Returns:
        jnp.array: Updated state
    """
    # Matrix multiplication for weight calculation
    weighted_sums = jnp.dot(graph_matrix, state)

    # Apply threshold activation function
    new_state = jnp.greater_equal(weighted_sums, 0).astype(jnp.int32)

    return new_state


def jax_update_batch_states(states, graph_matrix):
    """
    JAX-optimized batched state update using vmap.

    Args:
        states: Batch of states (jnp.array)
        graph_matrix: Connection graph matrix (jnp.array)

    Returns:
        jnp.array: Batch of updated states
    """
    # Vectorize the single state update function across the first dimension (batch)
    batch_update_fn = vmap(jax_update_single_state, in_axes=(0, None))
    return batch_update_fn(states, graph_matrix)


def process_states_in_batches(states, graph_matrix, batch_size=64, max_iterations=50, device=None):
    """
    Process states in batches using JAX optimization.

    Args:
        states: List of initial states
        graph_matrix: Connection graph matrix
        batch_size: Size of batches to process at once
        max_iterations: Maximum number of iterations
        device: JAX device to use for computation

    Returns:
        Dict: Mapping of initial states to final states
    """
    # Convert inputs to JAX arrays
    graph_matrix_jax = jnp.array(graph_matrix, dtype=jnp.int32)

    # Results dictionary
    results = {}

    # Process states in batches
    for i in range(0, len(states), batch_size):
        batch_states = states[i : i + batch_size]

        # Convert batch to JAX array
        batch_states_jax = jnp.array(batch_states, dtype=jnp.int32)

        # Transfer to device if specified
        if device is not None:
            batch_states_jax = device_put(batch_states_jax, device)

        # Initialize for iteration tracking
        curr_states = batch_states_jax
        iterations = 0

        # Track state history for cycle detection
        state_history = {}

        while iterations < max_iterations:
            # Update all states in the batch
            new_states = jax_update_batch_states(curr_states, graph_matrix_jax)

            # Check if states have converged
            all_equal = jnp.all(new_states == curr_states)

            # If all states are stable, we're done
            if all_equal:
                break

            curr_states = new_states
            iterations += 1

        # Convert final states back to lists for consistency
        final_states = jnp.array(curr_states).tolist()

        # Store results
        for init_state, final_state in zip(batch_states, final_states):
            init_tuple = tuple(init_state)
            final_tuple = tuple(final_state)
            results[init_tuple] = final_tuple

    return results


def configure_jax_devices(use_gpu=True, use_cpu=True, force_cpu_only=False):
    """
    Configure JAX devices based on user preferences and hardware availability.

    Args:
        use_gpu: Whether to use GPU devices if available
        use_cpu: Whether to use CPU devices
        force_cpu_only: Whether to force CPU-only mode even if GPUs are available

    Returns:
        Tuple[List, List]: Lists of available GPU and CPU devices
    """
    # Initialize empty device lists
    gpu_devices = []
    cpu_devices = []

    # Get all available devices
    all_devices = jax.devices()

    # Separate GPU and CPU devices
    for device in all_devices:
        if device.platform == "gpu" and use_gpu and not force_cpu_only:
            gpu_devices.append(device)
        elif device.platform == "cpu" and use_cpu:
            cpu_devices.append(device)

    return gpu_devices, cpu_devices


def setup_primary_device(gpu_devices, cpu_devices, force_cpu_only=False):
    """
    Set up the primary computation device based on availability and user preferences.

    Args:
        gpu_devices: List of available GPU devices
        cpu_devices: List of available CPU devices
        force_cpu_only: Whether to force CPU-only mode

    Returns:
        jax.Device: The primary JAX device for computation
    """
    # Use GPU if available and requested
    if gpu_devices and not force_cpu_only:
        primary_device = gpu_devices[0]  # Use first GPU
    # Otherwise use CPU
    elif cpu_devices:
        primary_device = cpu_devices[0]  # Use first CPU
    else:
        # Default to CPU if no devices were properly configured
        primary_device = jax.devices("cpu")[0]

    return primary_device


def calculate_optimal_batch_size(cyclins_count, batch_size_gpu=None, gpu_devices=None, force_cpu_only=False):
    """
    Calculate optimal batch size for GPU computation based on model complexity.

    Args:
        cyclins_count: Number of cyclins in the model
        batch_size_gpu: User-specified batch size (takes precedence if provided)
        gpu_devices: List of available GPU devices
        force_cpu_only: Whether CPU-only mode is forced

    Returns:
        int: Optimal batch size for computation
    """
    # If user provided a batch size, use that
    if batch_size_gpu is not None:
        return batch_size_gpu

    # If no GPU available or force CPU only, return a small default batch size
    if not gpu_devices or force_cpu_only:
        return 32  # Small default for CPU

    try:
        # Calculate states complexity
        num_states = 2**cyclins_count

        # Adjust batch size based on model complexity
        if cyclins_count <= 8:
            batch_size = 512
        elif cyclins_count <= 10:
            batch_size = 256
        elif cyclins_count <= 12:
            batch_size = 128
        elif cyclins_count <= 14:
            batch_size = 64
        else:
            batch_size = 32

        # For very large problems, limit even further
        if num_states > 1_000_000:  # More than ~20 cyclins
            batch_size = min(batch_size, 16)

        return batch_size

    except Exception:
        return 64  # Reasonable default
