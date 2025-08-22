import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.all_inputs import ModelAInputs, ModelBInputs, ModelCInputs
from optimized.state_calc import CellCycleStateCalculator
from optimized.utils import get_perturbations


def run_simulation_for_graph(args):
    graph_matrix, graph_id, file_inputs, model_inputs, user_inputs = args

    state_calc_obj = CellCycleStateCalculator(file_inputs, model_inputs, user_inputs)

    graph_score, final_state_count, state_seq_types = state_calc_obj.generate_graph_score_and_final_states(
        graph_matrix=np.array(graph_matrix, dtype=np.int_), graph_mod_id=graph_id
    )

    return graph_score.item(), final_state_count, state_seq_types


def main():
    parser = argparse.ArgumentParser(description="Optimized Cell Cycle Simulation")
    parser.add_argument("--organism", type=str, default="model01", help="Model to simulate")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per graph")
    parser.add_argument("--perturbation_count", type=int, default=0, help="Number of perturbations to apply")
    args = parser.parse_args()

    input_models = {"model01": ModelAInputs(), "model02": ModelBInputs(), "model03": ModelCInputs()}
    model_inputs = input_models.get(args.organism)

    # Load simulation parameters from YAML (assuming a default path)
    sim_params_path = Path("sim_input", "simulation_params.yaml")
    with open(sim_params_path, "r") as f:
        file_inputs = yaml.safe_load(f)

    original_graph = model_inputs.modified_graph

    # Generate perturbed graphs
    if args.perturbation_count > 0:
        perturbed_graphs_info = get_perturbations(np.array(original_graph, dtype=np.int_), args.perturbation_count)
    else:
        perturbed_graphs_info = [(original_graph, "Original Graph")]

    all_results = []

    # Prepare arguments for multiprocessing
    mp_args = []
    for graph_matrix, graph_id in perturbed_graphs_info:
        mp_args.append((graph_matrix, graph_id, file_inputs, model_inputs, args))

    # Run simulations in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_simulation_for_graph, mp_args)

    for (graph_matrix, graph_id), (avg_score, final_state_sum, state_seq_type) in zip(perturbed_graphs_info, results):
        all_results.append({
            "Graph ID": graph_id,
            "Average Score": avg_score,
            "Final State Sum": final_state_sum,
            "State Sequence Type": state_seq_type
        })

    # Output results (e.g., print or save to CSV)
    for result in all_results:
        print(f"Graph ID: {result['Graph ID']}, Avg Score: {result['Average Score']}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
