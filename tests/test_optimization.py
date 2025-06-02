#!/usr/bin/env python3
"""
Quick test script to verify GPU batch optimization functionality
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from argparse import Namespace

import jax
import jax.numpy as jnp

from src.all_inputs import ModelAInputs
from src.state_calc import CellCycleStateCalculation


def test_gpu_batch_optimization():
    """Test GPU batch processing optimization"""

    print("=== GPU Batch Optimization Test ===")

    # Test configuration
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

    # Create calculator instance
    calc = CellCycleStateCalculation(file_inputs, model_inputs, user_inputs)

    print(f"JAX Platform: {jax.default_backend()}")
    print(f"GPU Devices Available: {len(calc.gpu_devices)}")
    print(f"CPU Devices Available: {len(calc.cpu_devices)}")
    print(f"Optimal Batch Size: {calc.batch_size_gpu}")
    print(f"Primary Device: {calc.primary_device}")

    # Set up a simple test graph
    num_cyclins = len(calc._all_cyclins)
    test_graph = jnp.ones((num_cyclins, num_cyclins))  # Simple all-connected graph
    calc.set_custom_connected_graph(test_graph.tolist(), "TestGraph")

    # Test state iteration
    print(f"Number of Starting States: {len(calc._start_states)}")
    print(f"Number of G1 States: {len(calc._g1_start_states)}")

    # Run a small test
    try:
        graph_score, final_state_count, state_seq_types = calc.generate_graph_score_and_final_states()
        print(f"Graph Score: {graph_score}")
        print(f"Number of Final States: {len(final_state_count)}")
        print("✅ GPU batch processing test completed successfully!")
        return True
    except Exception as e:
        print(f"❌ GPU batch processing test failed: {e}")
        return False


def test_batch_size_optimization():
    """Test dynamic batch size calculation"""

    print("\n=== Batch Size Optimization Test ===")

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

    # Test with different scenarios
    scenarios = [
        {"batch_size_gpu": None, "desc": "Auto-detected"},
        {"batch_size_gpu": 32, "desc": "User-specified 32"},
        {"batch_size_gpu": 128, "desc": "User-specified 128"},
        {"force_cpu_only": True, "desc": "CPU-only mode"},
    ]

    for scenario in scenarios:
        test_inputs = {**file_inputs, **scenario}
        calc = CellCycleStateCalculation(test_inputs, model_inputs, user_inputs)
        print(f"{scenario['desc']}: Batch size = {calc.batch_size_gpu}")

    print("✅ Batch size optimization test completed!")
    return True


if __name__ == "__main__":
    success = True
    success &= test_gpu_batch_optimization()
    success &= test_batch_size_optimization()

    if success:
        print("\n🎉 All optimization tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
