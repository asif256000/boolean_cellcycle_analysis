#!/usr/bin/env python3
"""
Benchmark for memory-optimized JAX acceleration on large models.

This script tests the memory optimization techniques for processing
very large Boolean cell cycle models that would normally exceed GPU memory.
"""

import logging
import os
import sys
import time
from argparse import Namespace

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.all_inputs import ModelAInputs
from src.jax_accelerator import JAX_AVAILABLE
from src.state_calc import CellCycleStateCalculation

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("memory_benchmark")

# Only import JAX-related modules if available
if JAX_AVAILABLE:
    try:
        from src.memory_optimized_calc import MemoryOptimizedCalculation

        MEMORY_OPTIMIZER_AVAILABLE = True
    except ImportError:
        MEMORY_OPTIMIZER_AVAILABLE = False
        logger.warning("Memory-optimized calculator not available")

    from src.jax_state_calc import JaxCellCycleCalculation


def benchmark_large_model():
    """Benchmark performance on a large model with memory constraints"""
    logger.info("=== Large Model Memory Optimization Benchmark ===")

    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping benchmark.")
        return

    if not MEMORY_OPTIMIZER_AVAILABLE:
        logger.warning("Memory optimization is not available. Skipping benchmark.")
        return

    # Standard configuration
    file_inputs = {
        "hardcoded_self_loops": True,
        "check_sequence": False,
        "g1_states_only": True,  # Only use G1 states to reduce workload
        "async_update": False,
        "random_order_cyclin": False,
        "complete_cycle": False,
        "expensive_state_cycle_detection": False,
        "max_updates_per_cycle": 50,
        "use_gpu": True,
        "use_cpu": True,
        "force_cpu_only": False,
        "use_memory_optimization": True,
        "memory_limit_gb": 2.0,  # Simulate a constrained environment
    }

    model_inputs = ModelAInputs()
    user_inputs = Namespace(organism="ModelA")

    # Create a test graph - we'll use a smaller one for testing
    num_cyclins = len(model_inputs.cyclins)
    test_graph = [[1 for _ in range(num_cyclins)] for _ in range(num_cyclins)]

    # Test configurations
    configs = [
        {"name": "Standard JAX", "use_memory_opt": False},
        {"name": "Memory-Optimized JAX", "use_memory_opt": True},
    ]

    results = {}

    for config in configs:
        logger.info(f"\n--- Testing {config['name']} ---")

        # Create appropriate calculator
        if config["use_memory_opt"]:
            calc = MemoryOptimizedCalculation(file_inputs, model_inputs, user_inputs)
        else:
            calc = JaxCellCycleCalculation(file_inputs, model_inputs, user_inputs)

        calc.set_custom_connected_graph(test_graph, "TestGraph")

        # Run calculation and time it
        start_time = time.time()
        try:
            logger.info("Starting graph score calculation...")
            score, states, _ = calc.generate_graph_score_and_final_states()
            exec_time = time.time() - start_time

            logger.info(f"Successfully completed in {exec_time:.4f} seconds")
            logger.info(f"Graph Score: {score}")
            logger.info(f"Final States: {len(states)}")

            results[config["name"]] = {"time": exec_time, "success": True, "score": score, "states": len(states)}

        except Exception as e:
            exec_time = time.time() - start_time
            logger.error(f"Failed after {exec_time:.4f} seconds: {str(e)}")
            results[config["name"]] = {"time": exec_time, "success": False, "error": str(e)}

    # Compare results
    logger.info("\n=== Results Comparison ===")
    for name, result in results.items():
        if result["success"]:
            logger.info(
                f"{name}: ✅ Success - {result['time']:.4f}s - Score: {result['score']}, States: {result['states']}"
            )
        else:
            logger.info(f"{name}: ❌ Failed - {result['time']:.4f}s - Error: {result['error']}")

    # Compare performance if both succeeded
    if all(result["success"] for result in results.values()):
        std_time = results["Standard JAX"]["time"]
        opt_time = results["Memory-Optimized JAX"]["time"]

        if std_time > opt_time:
            speedup = std_time / opt_time
            logger.info(f"Memory-optimized version is {speedup:.2f}x faster!")
        else:
            slowdown = opt_time / std_time
            logger.info(f"Memory-optimized version is {slowdown:.2f}x slower.")


if __name__ == "__main__":
    benchmark_large_model()
