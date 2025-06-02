#!/usr/bin/env python3
"""
Test the batch size optimization logic for JAX acceleration.

This test validates that:
1. The optimal batch size calculation works correctly for different model sizes
2. Batch size is properly limited for very large models
3. The system respects user-provided batch sizes when specified
4. The system gracefully handles errors and edge cases
"""

import logging
import os
import sys
import time
from argparse import Namespace

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.jax_accelerator import JAX_AVAILABLE, calculate_optimal_batch_size

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("batch_size_test")


def test_batch_size_calculation():
    """Test batch size calculation logic for different scenarios"""
    logger.info("=== Testing Batch Size Optimization Logic ===")

    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping batch size tests.")
        return

    # Test for different model complexities
    test_cases = [
        {"cyclins": 6, "expected_range": (256, 512)},
        {"cyclins": 8, "expected_range": (256, 512)},
        {"cyclins": 10, "expected_range": (128, 256)},
        {"cyclins": 12, "expected_range": (64, 128)},
        {"cyclins": 14, "expected_range": (32, 64)},
        {"cyclins": 16, "expected_range": (16, 32)},
        {"cyclins": 20, "expected_range": (8, 32)},
        {"cyclins": 24, "expected_range": (4, 16)},
    ]

    for case in test_cases:
        batch_size = calculate_optimal_batch_size(case["cyclins"])
        min_expected, max_expected = case["expected_range"]

        if min_expected <= batch_size <= max_expected:
            logger.info(
                f"✅ Model with {case['cyclins']} cyclins → Batch size {batch_size} is within expected range {case['expected_range']}"
            )
        else:
            logger.error(
                f"❌ Model with {case['cyclins']} cyclins → Batch size {batch_size} is outside expected range {case['expected_range']}"
            )


def test_user_override():
    """Test that user-provided batch sizes are respected"""
    logger.info("\n=== Testing User Override of Batch Size ===")

    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping batch size tests.")
        return

    # Test cases where user specifies a batch size
    test_cases = [
        {"cyclins": 8, "user_batch": 64, "expected": 64},
        {"cyclins": 12, "user_batch": 256, "expected": 256},
        {"cyclins": 16, "user_batch": 32, "expected": 32},
    ]

    for case in test_cases:
        batch_size = calculate_optimal_batch_size(case["cyclins"], batch_size_gpu=case["user_batch"])

        if batch_size == case["expected"]:
            logger.info(f"✅ User-specified batch size {case['user_batch']} was correctly used")
        else:
            logger.error(f"❌ User-specified batch size {case['user_batch']} was overridden with {batch_size}")


def test_force_cpu_only():
    """Test that force_cpu_only parameter works correctly"""
    logger.info("\n=== Testing Force CPU Only Mode ===")

    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping CPU-only mode tests.")
        return

    # Standard model size
    num_cyclins = 10

    # Compare normal vs CPU-only batch sizes
    normal_batch = calculate_optimal_batch_size(num_cyclins)
    cpu_batch = calculate_optimal_batch_size(num_cyclins, force_cpu_only=True)

    logger.info(f"Normal batch size: {normal_batch}")
    logger.info(f"CPU-only batch size: {cpu_batch}")

    # CPU-only should use more conservative batch sizes
    if cpu_batch <= normal_batch:
        logger.info("✅ CPU-only mode correctly uses smaller or equal batch size")
    else:
        logger.error("❌ CPU-only mode unexpectedly uses larger batch size")


def test_edge_cases():
    """Test edge cases and error handling"""
    logger.info("\n=== Testing Edge Cases ===")

    if not JAX_AVAILABLE:
        logger.warning("JAX is not available. Skipping edge case tests.")
        return

    # Edge cases to test
    edge_cases = [
        {"desc": "Very small model", "cyclins": 2},
        {"desc": "Very large model", "cyclins": 30},
        {"desc": "Negative cyclins", "cyclins": -5},
        {"desc": "Zero cyclins", "cyclins": 0},
    ]

    for case in edge_cases:
        try:
            batch_size = calculate_optimal_batch_size(case["cyclins"])
            logger.info(f"{case['desc']} (cyclins={case['cyclins']}) → Batch size: {batch_size}")

            # Additional validation for extreme values
            if batch_size <= 0:
                logger.error(f"❌ Invalid batch size: {batch_size} (must be positive)")
            elif batch_size > 1024:
                logger.warning(f"⚠️ Very large batch size: {batch_size} (may cause memory issues)")

        except Exception as e:
            logger.error(f"❌ Error calculating batch size for {case['desc']}: {str(e)}")


if __name__ == "__main__":
    test_batch_size_calculation()
    test_user_override()
    test_force_cpu_only()
    test_edge_cases()
