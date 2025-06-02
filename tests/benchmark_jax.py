"""
Test and benchmark the JAX-accelerated state calculation.

This script compares the original and JAX-accelerated implementations
to verify correctness and demonstrate performance improvements.
"""

import os
import sys
import time
from argparse import Namespace

import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.all_inputs import ModelAInputs
from src.jax_accelerator import JAX_AVAILABLE
from src.state_calc import CellCycleStateCalculation

# Only import if JAX is available
if JAX_AVAILABLE:
    from src.jax_state_calc import JaxCellCycleCalculation


def compare_implementations():
    """
    Compare the original and JAX-accelerated implementations.

    This function runs both implementations on the same input data
    and compares the results for correctness and performance.
    """
    print("=== Comparing CellCycleStateCalculation Implementations ===")

    # Standard configuration for all tests
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

    # Create test graph - a simple all-connected graph
    num_cyclins = len(model_inputs.cyclins)
    test_graph = [[1 for _ in range(num_cyclins)] for _ in range(num_cyclins)]

    # Test original implementation
    print("\n--- Original Implementation ---")
    orig_calc = CellCycleStateCalculation(file_inputs, model_inputs, user_inputs)
    orig_calc.set_custom_connected_graph(test_graph, "OriginalTestGraph")

    start_time = time.time()
    orig_score, orig_states, orig_seq = orig_calc.generate_graph_score_and_final_states()
    orig_time = time.time() - start_time

    print(f"Graph Score: {orig_score}")
    print(f"Number of Final States: {len(orig_states)}")
    print(f"Execution Time: {orig_time:.4f} seconds")

    # Test JAX implementation if available
    if JAX_AVAILABLE:
        print("\n--- JAX-Accelerated Implementation ---")
        jax_calc = JaxCellCycleCalculation(file_inputs, model_inputs, user_inputs)
        jax_calc.set_custom_connected_graph(test_graph, "JaxTestGraph")

        start_time = time.time()
        jax_score, jax_states, jax_seq = jax_calc.generate_graph_score_and_final_states()
        jax_time = time.time() - start_time

        print(f"Graph Score: {jax_score}")
        print(f"Number of Final States: {len(jax_states)}")
        print(f"Execution Time: {jax_time:.4f} seconds")

        # Compare results
        print("\n--- Comparison ---")
        if orig_score == jax_score:
            print("✅ Graph scores match!")
        else:
            print(f"❌ Graph scores differ: Original={orig_score}, JAX={jax_score}")

        if len(orig_states) == len(jax_states):
            print("✅ Number of final states match!")
        else:
            print(f"❌ Number of final states differ: Original={len(orig_states)}, JAX={len(jax_states)}")

        if jax_time < orig_time:
            speedup = orig_time / jax_time
            print(f"✅ JAX implementation is {speedup:.2f}x faster!")
        else:
            slowdown = jax_time / orig_time
            print(f"❌ JAX implementation is {slowdown:.2f}x slower.")
    else:
        print("\nJAX acceleration is not available. Only ran original implementation.")


def benchmark_batch_sizes():
    """Benchmark different batch sizes to find the optimal configuration"""
    if not JAX_AVAILABLE:
        print("JAX is not available, skipping batch size benchmark")
        return

    print("\n=== Batch Size Optimization Benchmark ===")

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
        print(f"\nTesting batch size: {batch_size}")

        # Create calculator with specific batch size
        test_inputs = {**file_inputs, "batch_size_gpu": batch_size}
        jax_calc = JaxCellCycleCalculation(test_inputs, model_inputs, user_inputs)
        jax_calc.set_custom_connected_graph(test_graph, f"BatchSize{batch_size}")

        start_time = time.time()
        score, states, _ = jax_calc.generate_graph_score_and_final_states()
        exec_time = time.time() - start_time

        print(f"Execution Time: {exec_time:.4f} seconds")
        results[batch_size] = exec_time

    # Find optimal batch size
    optimal_batch = min(results, key=results.get)
    print(f"\nOptimal batch size: {optimal_batch} (Time: {results[optimal_batch]:.4f}s)")

    # Print all results
    print("\nAll batch size results:")
    for batch_size, exec_time in sorted(results.items()):
        print(f"Batch size {batch_size}: {exec_time:.4f}s")


if __name__ == "__main__":
    compare_implementations()
    benchmark_batch_sizes()
