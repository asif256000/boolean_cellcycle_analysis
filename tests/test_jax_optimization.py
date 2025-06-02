"""
Test the JAX accelerated state calculation in the boolean cell cycle analysis.

This test script verifies that:
1. JAX acceleration works correctly and produces the same results as the original implementation
2. Optimal batch size calculation works for different model complexities
3. Device management (GPU vs CPU) works correctly
4. The implementation gracefully handles JAX unavailability
"""

import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
from argparse import Namespace

import numpy as np

from src.all_inputs import ModelAInputs
from src.jax_accelerator import JAX_AVAILABLE, calculate_optimal_batch_size
from src.state_calc import CellCycleStateCalculation

# Only import JAX-related modules if available
if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    from src.jax_state_calc import JaxCellCycleCalculation

# Configure logging
logging.basicConfig(level=logging.INFO)


def test_jax_optimization():
    """Test JAX state calculation optimization"""
    print("=== JAX State Calculation Optimization Test ===")

    if not JAX_AVAILABLE:
        print("JAX is not available. Skipping JAX optimization tests.")
        return

    # Test with GPU acceleration if available
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Available Devices: {jax.devices()}")

    # Standard config for all tests
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

    # Test configurations
    configs = [
        {"name": "Original Implementation", "jax_accel": False},
        {"name": "JAX-Accelerated (Auto)", "jax_accel": True, "force_cpu_only": False},
        {"name": "JAX-Accelerated (CPU-only)", "jax_accel": True, "force_cpu_only": True},
    ]

    results = {}

    # Run tests with different configurations
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")

        # Update config for this test
        test_inputs = file_inputs.copy()
        if config["jax_accel"] and "force_cpu_only" in config:
            test_inputs["force_cpu_only"] = config["force_cpu_only"]

        # Create appropriate calculator instance
        if config["jax_accel"]:
            calc = JaxCellCycleCalculation(test_inputs, model_inputs, user_inputs)
        else:
            calc = CellCycleStateCalculation(test_inputs, model_inputs, user_inputs)

        # Set up test graph - simple all-connected graph
        num_cyclins = len(model_inputs.cyclins)
        test_graph = [[1 for _ in range(num_cyclins)] for _ in range(num_cyclins)]
        calc.set_custom_connected_graph(test_graph, "TestGraph")

        # Run the calculation and time it
        start_time = time.time()
        graph_score, final_states, state_seq_types = calc.generate_graph_score_and_final_states()
        elapsed = time.time() - start_time

        print(f"Graph Score: {graph_score}")
        print(f"Number of Final States: {len(final_states)}")
        print(f"Execution Time: {elapsed:.4f} seconds")

        # Store results for comparison
        results[config["name"]] = {"score": graph_score, "states": final_states, "time": elapsed}

    # Compare results for correctness
    print("\n--- Comparison ---")
    base_result = results["Original Implementation"]

    for name, result in results.items():
        if name == "Original Implementation":
            continue

        if result["score"] == base_result["score"]:
            print(f"✅ {name}: Graph scores match!")
        else:
            print(f"❌ {name}: Graph scores differ: Original={base_result['score']}, {name}={result['score']}")

        if len(result["states"]) == len(base_result["states"]):
            print(f"✅ {name}: Number of final states match!")
        else:
            print(
                f"❌ {name}: Number of states differ: Original={len(base_result['states'])}, {name}={len(result['states'])}"
            )

        speedup = base_result["time"] / result["time"]
        if speedup > 1:
            print(f"✅ {name}: {speedup:.2f}x faster than original!")
        else:
            print(f"❌ {name}: {1 / speedup:.2f}x slower than original.")


def test_batch_size_settings():
    """Test different batch size configurations"""
    print("\n=== Batch Size Configuration Test ===")

    if not JAX_AVAILABLE:
        print("JAX is not available. Skipping batch size tests.")
        return

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
    }

    model_inputs = ModelAInputs()
    user_inputs = Namespace(organism="ModelA")

    # Test different batch sizes
    batch_sizes = [None, 32, 64, 128, 256]

    for batch_size in batch_sizes:
        test_inputs = {**file_inputs, "batch_size_gpu": batch_size}
        calc = JaxCellCycleCalculation(test_inputs, model_inputs, user_inputs)
        print(f"Configured batch size: {batch_size}, Actual batch size: {calc.batch_size_gpu}")

    # Test optimal batch size calculation for different model complexities
    print("\n--- Testing Optimal Batch Size Calculation ---")
    cyclin_counts = [6, 8, 10, 12, 14, 16, 20]

    for num_cyclins in cyclin_counts:
        optimal_batch = calculate_optimal_batch_size(num_cyclins)
        print(f"Model with {num_cyclins} cyclins → Optimal batch size: {optimal_batch}")

    # Test large model batch size limits
    large_model = 24  # 2^24 states is a very large state space
    limited_batch = calculate_optimal_batch_size(large_model)
    print(f"Large model ({large_model} cyclins) → Limited batch size: {limited_batch}")


def test_device_management():
    """Test device management and configuration"""
    if not JAX_AVAILABLE:
        print("\n=== Device Management Test ===")
        print("JAX is not available. Skipping device management tests.")
        return

    print("\n=== Device Management Test ===")

    # Test configurations
    configs = [
        {"name": "Auto (GPU if available)", "use_gpu": True, "use_cpu": True, "force_cpu_only": False},
        {"name": "Force CPU Only", "use_gpu": True, "use_cpu": True, "force_cpu_only": True},
        {"name": "GPU Disabled", "use_gpu": False, "use_cpu": True, "force_cpu_only": False},
    ]

    for config in configs:
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

        print(f"\n--- {config['name']} ---")
        calc = JaxCellCycleCalculation(file_inputs, model_inputs, user_inputs)

        # Display selected device info
        print(f"GPU devices: {len(calc.gpu_devices)}")
        print(f"CPU devices: {len(calc.cpu_devices)}")
        print(f"Primary device: {calc.primary_device}")


if __name__ == "__main__":
    test_jax_optimization()
    test_batch_size_settings()
    test_device_management()
