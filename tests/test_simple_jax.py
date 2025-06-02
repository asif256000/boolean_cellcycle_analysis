"""
A minimal example to test JAX-accelerated state calculations.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import device_put, jit, vmap

# Use CPU for simplicity in testing
jax.config.update("jax_platform_name", "cpu")


def simple_update_function(state, graph_matrix):
    """Pure implementation of state update without class dependencies"""
    weighted_sums = jnp.dot(graph_matrix, state)
    new_state = jnp.greater_equal(weighted_sums, 0).astype(jnp.int32)
    return new_state


# Vectorize the update function
batch_update = vmap(simple_update_function, in_axes=(0, None))

# JIT-compile the batch function for better performance
batch_update_jit = jit(batch_update)


def process_states(states, graph_matrix, max_iterations=50, batch_size=64):
    """Process states in batches using JAX"""
    # Convert to JAX arrays
    graph_matrix_jax = jnp.array(graph_matrix, dtype=jnp.int32)
    results = {}

    # Process in batches
    for i in range(0, len(states), batch_size):
        batch = states[i : i + batch_size]
        batch_jax = jnp.array(batch, dtype=jnp.int32)

        # Iterate until convergence
        curr_states = batch_jax
        iterations = 0

        while iterations < max_iterations:
            new_states = batch_update_jit(curr_states, graph_matrix_jax)
            all_equal = jnp.all(new_states == curr_states)

            if all_equal:
                break

            curr_states = new_states
            iterations += 1

        # Store results
        final_states = jnp.array(curr_states).tolist()
        for init_state, final_state in zip(batch, final_states):
            results[tuple(init_state)] = tuple(final_state)

    return results


def run_simple_test():
    """Run a simple test of the JAX acceleration"""
    print("=== JAX Acceleration Simple Test ===")
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Available Devices: {jax.devices()}")

    # Create a simple test graph (all connections are 1)
    num_nodes = 5
    graph = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Generate all possible states for this graph
    all_states = []
    for i in range(2**num_nodes):
        binary = format(i, f"0{num_nodes}b")
        state = [int(b) for b in binary]
        all_states.append(state)

    print(f"Processing {len(all_states)} states...")

    # Time the JAX processing
    start_time = time.time()
    results = process_states(all_states, graph, max_iterations=50)
    elapsed = time.time() - start_time

    print(f"JAX processing completed in {elapsed:.4f} seconds")
    print(f"Number of unique final states: {len(set(results.values()))}")

    # Sample results
    sample_keys = list(results.keys())[:3]
    for key in sample_keys:
        print(f"Initial state: {key} → Final state: {results[key]}")

    return True


if __name__ == "__main__":
    run_simple_test()
