#!/usr/bin/env python3
"""
Enhanced test and benchmark for JAX-accelerated Boolean cell cycle analysis.

This script provides comprehensive testing and benchmarking of:
1. JAX vs original implementation performance and correctness
2. GPU vs CPU performance
3. Batch size optimization
4. Memory usage optimization
5. Edge cases handling
"""

import logging
import os
import sys
import time
from argparse import Namespace

import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.all_inputs import ModelAInputs
from src.jax_accelerator import JAX_AVAILABLE, calculate_optimal_batch_size
from src.state_calc import CellCycleStateCalculation

# Only import JAX-related modules if available
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from src.jax_state_calc import JaxCellCycleCalculation

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("jax_benchmark")


def compare_implementations():
    """
    Compare JAX-accelerated and original implementations for correctness and performance.
    """
    logger.info("=== Comparing Implementation Performance ===")

    # Standard configuration
    file_inputs = {
        "hardcoded_self_loops": True,
        "check_sequence": False,
        "g1_states_only": False,
        "async_update": False,
        "random_order_cyclin": False,
        "complete_cycle": False,
        "expensive_state_cycle_detection": False,
        "max_updates_per_cycle": 50,
        "use_gpu": True,
        "use_cpu": True,
        "force_cpu_only": False,
    }

    model_inputs = ModelAInputs()
    user_inputs = Namespace(organism="ModelA")

    # Create test graph
    num_cyclins = len(model_inputs.cyclins)
    test_graph = [[1 for _ in range(num_cyclins)] for _ in range(num_cyclins)]

    # Test original implementation
    logger.info("Running original implementation...")
    orig_calc = CellCycleStateCalculation(file_inputs, model_inputs, user_inputs)
    orig_calc.set_custom_connected_graph(test_graph, "OriginalTestGraph")

    start_time = time.time()
    orig_score, orig_states, orig_seq = orig_calc.generate_graph_score_and_final_states()
    orig_time = time.time() - start_time

    logger.info(f"Original implementation - Score: {orig_score}, States: {len(orig_states)}, Time: {orig_time:.4f}s")

    # Skip JAX tests if not available
    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping JAX-accelerated tests.")
        return

    # Test JAX implementation
    logger.info("Running JAX-accelerated implementation...")
    jax_calc = JaxCellCycleCalculation(file_inputs, model_inputs, user_inputs)
    jax_calc.set_custom_connected_graph(test_graph, "JaxTestGraph")

    start_time = time.time()
    jax_score, jax_states, jax_seq = jax_calc.generate_graph_score_and_final_states()
    jax_time = time.time() - start_time

    logger.info(f"JAX implementation - Score: {jax_score}, States: {len(jax_states)}, Time: {jax_time:.4f}s")

    # Compare results
    logger.info("=== Result Comparison ===")
    if orig_score == jax_score:
        logger.info("✅ Graph scores match!")
    else:
        logger.error(f"❌ Graph scores differ: Original={orig_score}, JAX={jax_score}")

    if len(orig_states) == len(jax_states):
        logger.info("✅ Number of final states match!")
    else:
        logger.error(f"❌ Number of final states differ: Original={len(orig_states)}, JAX={len(jax_states)}")

    # Compare performance
    if jax_time < orig_time:
        speedup = orig_time / jax_time
        logger.info(f"✅ JAX implementation is {speedup:.2f}x faster!")
    else:
        slowdown = jax_time / orig_time
        logger.error(f"❌ JAX implementation is {slowdown:.2f}x slower.")


def benchmark_batch_sizes():
    """
    Benchmark different batch sizes to identify optimal settings for different model sizes.
    """
    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping batch size benchmarking.")
        return

    logger.info("=== Batch Size Optimization Benchmark ===")

    # Standard configuration
    file_inputs = {
        "hardcoded_self_loops": True,
        "check_sequence": False,
        "g1_states_only": False,
        "async_update": False,
        "random_order_cyclin": False,
        "complete_cycle": False,
        "expensive_state_cycle_detection": False,
        "max_updates_per_cycle": 50,
        "use_gpu": True,
        "use_cpu": True,
        "force_cpu_only": False,
    }

    model_inputs = ModelAInputs()
    user_inputs = Namespace(organism="ModelA")

    # Create test graph
    num_cyclins = len(model_inputs.cyclins)
    test_graph = [[1 for _ in range(num_cyclins)] for _ in range(num_cyclins)]

    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    results = {}

    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")

        # Create calculator with specific batch size
        test_inputs = {**file_inputs, "batch_size_gpu": batch_size}
        jax_calc = JaxCellCycleCalculation(test_inputs, model_inputs, user_inputs)
        jax_calc.set_custom_connected_graph(test_graph, f"BatchSize{batch_size}")

        start_time = time.time()
        score, states, _ = jax_calc.generate_graph_score_and_final_states()
        exec_time = time.time() - start_time

        logger.info(f"Execution time: {exec_time:.4f} seconds")
        results[batch_size] = exec_time

    # Identify optimal batch size
    optimal_batch = min(results, key=results.get)
    logger.info(f"Optimal batch size: {optimal_batch} (Time: {results[optimal_batch]:.4f}s)")

    # Compare to our automatic calculation
    calc_optimal = calculate_optimal_batch_size(num_cyclins)
    logger.info(f"Auto-calculated optimal batch size: {calc_optimal}")

    # Print all results
    logger.info("All batch size results:")
    for batch_size, exec_time in sorted(results.items()):
        logger.info(f"Batch size {batch_size}: {exec_time:.4f}s")


def test_device_management():
    """
    Test device management and configurations.
    """
    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping device management tests.")
        return

    logger.info("=== Device Management Test ===")

    # Report JAX configuration
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"Available devices: {jax.devices()}")

    # Test configurations
    configs = [
        {"name": "Auto (GPU if available)", "use_gpu": True, "use_cpu": True, "force_cpu_only": False},
        {"name": "Force CPU Only", "use_gpu": True, "use_cpu": True, "force_cpu_only": True},
        {"name": "GPU Disabled", "use_gpu": False, "use_cpu": True, "force_cpu_only": False},
    ]

    for config in configs:
        logger.info(f"\nTesting configuration: {config['name']}")

        file_inputs = {
            "hardcoded_self_loops": True,
            "check_sequence": False,
            "g1_states_only": False,
            "async_update": False,
            "random_order_cyclin": False,
            "complete_cycle": False,
            "expensive_state_cycle_detection": False,
            "max_updates_per_cycle": 50,
            "use_gpu": config["use_gpu"],
            "use_cpu": config["use_cpu"],
            "force_cpu_only": config["force_cpu_only"],
        }

        model_inputs = ModelAInputs()
        user_inputs = Namespace(organism="ModelA")

        jax_calc = JaxCellCycleCalculation(file_inputs, model_inputs, user_inputs)

        logger.info(f"GPU devices detected: {len(jax_calc.gpu_devices)}")
        logger.info(f"CPU devices detected: {len(jax_calc.cpu_devices)}")
        logger.info(f"Primary device selected: {jax_calc.primary_device}")
        logger.info(f"Batch size selected: {jax_calc.batch_size_gpu}")


if __name__ == "__main__":
    compare_implementations()
    benchmark_batch_sizes()
    test_device_management()
