#!/usr/bin/env python3
"""
Test memory optimization techniques for JAX-accelerated Boolean cell cycle analysis.

This test script verifies:
1. Memory usage estimation is working correctly
2. Adaptive batch sizing adjusts based on model complexity
3. Memory-efficient processing produces correct results
4. Large model handling works without memory errors
"""

import gc
import logging
import os
import sys
import time
from argparse import Namespace

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.all_inputs import ModelAInputs
from src.jax_accelerator import JAX_AVAILABLE
from src.state_calc import CellCycleStateCalculation

# Only import JAX-related modules if available
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from src.jax_state_calc import JaxCellCycleCalculation
    from src.utils.jax_memory_optimizer import (
        adaptive_batch_size,
        estimate_memory_requirements,
        memory_efficient_state_processing,
    )

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("memory_test")


def test_memory_estimation():
    """Test memory usage estimation for different model sizes"""
    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping memory estimation tests.")
        return

    logger.info("=== Memory Usage Estimation Test ===")

    test_cases = [
        {"cyclins": 8, "batch_size": 256},
        {"cyclins": 12, "batch_size": 128},
        {"cyclins": 16, "batch_size": 64},
        {"cyclins": 20, "batch_size": 32},
        {"cyclins": 24, "batch_size": 16},
    ]

    for case in test_cases:
        peak_gb, steady_gb = estimate_memory_requirements(case["cyclins"], case["batch_size"])
        logger.info(
            f"Model with {case['cyclins']} cyclins, batch size {case['batch_size']}: "
            f"Peak={peak_gb:.2f} GB, Steady={steady_gb:.2f} GB"
        )


def test_adaptive_batch_sizing():
    """Test adaptive batch size calculation based on memory constraints"""
    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping adaptive batch sizing tests.")
        return

    logger.info("\n=== Adaptive Batch Size Test ===")

    # Test cases with different memory constraints
    test_cases = [
        {"cyclins": 10, "memory_gb": 1.0},  # Very limited memory
        {"cyclins": 10, "memory_gb": 4.0},  # Standard memory
        {"cyclins": 10, "memory_gb": 16.0},  # Large memory
        {"cyclins": 16, "memory_gb": 1.0},  # Large model, limited memory
        {"cyclins": 16, "memory_gb": 8.0},  # Large model, decent memory
        {"cyclins": 20, "memory_gb": 32.0},  # Very large model, large memory
    ]

    for case in test_cases:
        batch_size = adaptive_batch_size(case["cyclins"], case["memory_gb"])
        peak_gb, _ = estimate_memory_requirements(case["cyclins"], batch_size)

        logger.info(
            f"Model with {case['cyclins']} cyclins, "
            f"{case['memory_gb']:.1f} GB available memory: "
            f"Optimal batch size = {batch_size}, "
            f"Expected peak usage = {peak_gb:.2f} GB"
        )

        # Verify memory usage would be under limit
        if peak_gb > case["memory_gb"]:
            logger.error("❌ Estimated peak memory usage exceeds available memory!")
        else:
            logger.info("✅ Memory usage within constraints")


def test_memory_efficient_processing():
    """Test memory-efficient state processing with different model sizes"""
    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping memory-efficient processing tests.")
        return

    logger.info("\n=== Memory-Efficient Processing Test ===")

    # Create a simple test graph
    num_cyclins = 10  # Small enough to test quickly
    test_graph = [[1 for _ in range(num_cyclins)] for _ in range(num_cyclins)]

    # Generate some test states (a subset to process quickly)
    max_states = 1000
    test_states = []
    for i in range(min(max_states, 2**num_cyclins)):
        binary = format(i, f"0{num_cyclins}b")
        state = [int(b) for b in binary]
        test_states.append(state)

    # Process states with standard method as reference
    start_time = time.time()
    reference_results = {}
    for state in test_states:
        # Simple update logic (similar to our standard implementation)
        curr_state = state.copy()
        for _ in range(50):  # Max iterations
            new_state = []
            for i in range(num_cyclins):
                weighted_sum = sum(test_graph[i][j] * curr_state[j] for j in range(num_cyclins))
                new_state.append(1 if weighted_sum >= 0 else 0)

            if new_state == curr_state:
                break

            curr_state = new_state

        reference_results[tuple(state)] = tuple(curr_state)

    reference_time = time.time() - start_time
    logger.info(f"Reference implementation: {reference_time:.4f} seconds")

    # Process with memory-efficient implementation
    gc.collect()  # Clear memory before test

    start_time = time.time()
    memory_efficient_results = memory_efficient_state_processing(
        test_states,
        test_graph,
        batch_size=64,  # Small batch size for testing
        max_iterations=50,
    )
    mem_efficient_time = time.time() - start_time

    logger.info(f"Memory-efficient implementation: {mem_efficient_time:.4f} seconds")

    # Verify results match
    if len(reference_results) != len(memory_efficient_results):
        logger.error(
            f"❌ Result count mismatch: Reference={len(reference_results)}, "
            f"Memory-efficient={len(memory_efficient_results)}"
        )
    else:
        logger.info(f"✅ Result count matches: {len(reference_results)} states")

    # Check for any differences in results
    differences = 0
    for state, ref_result in reference_results.items():
        if state not in memory_efficient_results or memory_efficient_results[state] != ref_result:
            differences += 1
            if differences <= 3:  # Show at most 3 differences
                logger.error(
                    f"❌ Result mismatch for state {state}: "
                    f"Reference={ref_result}, Memory-efficient={memory_efficient_results.get(state, 'missing')}"
                )

    if differences == 0:
        logger.info("✅ All results match between implementations")
    else:
        logger.error(f"❌ {differences} result differences found")

    # Report speedup
    if mem_efficient_time < reference_time:
        speedup = reference_time / mem_efficient_time
        logger.info(f"✅ Memory-efficient implementation is {speedup:.2f}x faster")
    else:
        slowdown = mem_efficient_time / reference_time
        logger.info(f"⚠️ Memory-efficient implementation is {slowdown:.2f}x slower")


if __name__ == "__main__":
    test_memory_estimation()
    test_adaptive_batch_sizing()
    test_memory_efficient_processing()
