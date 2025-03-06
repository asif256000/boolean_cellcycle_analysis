import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from all_inputs import ModelAInputs, ModelBInputs, ModelCInputs
from state_calc_clean import CellCycleStateCalculation
from utils import (
    all_perturbation_generator,
    draw_graph_from_matrix,
    parse_perturbation_string,
    single_perturbation_generator,
)

NPROC = None


def perturbation_mp_wrapper(args: tuple):
    state_calc_obj, graph, graph_mod_id, cyclins, iter_count = args
    graph_mod, avg_score, unique_final_states, max_state_avg_count, max_state, avg_seq = single_graph_execution(
        state_calc_obj=state_calc_obj,
        current_graph=graph,
        graph_mod=graph_mod_id,
        cyclins=cyclins,
        iterations=iter_count,
        mp=False,
    )
    return graph_mod, avg_score, unique_final_states, max_state_avg_count, max_state, avg_seq


def mp_wrapper(state_calc_obj: CellCycleStateCalculation):
    (
        graph_score,
        final_state_dict,
        state_seq_type,
    ) = state_calc_obj.generate_graph_score_and_final_states()
    return graph_score, final_state_dict, state_seq_type


def score_states_multiprocess(state_calc_obj: CellCycleStateCalculation, iter_count: int, multi_process: bool = True):
    """
    This function will run the given amount of iterations on the given graph and calculate the average score, correctness, and final states.
    """
    graph_score_sum = 0
    all_final_state_sum = dict()
    all_state_seq_type = dict()

    # Run the simulation
    if multi_process:
        with mp.Pool(processes=NPROC) as mp_pool:
            results = mp_pool.map(
                func=mp_wrapper,
                iterable=[state_calc_obj for _ in range(iter_count)],
            )
    else:
        results = list()
        for _ in range(iter_count):
            graph_score, final_state_dict, state_seq_type = state_calc_obj.generate_graph_score_and_final_states()
            results.append((graph_score, final_state_dict, state_seq_type))

    # Calculate statistics from the results
    for graph_score, final_state_dict, state_seq_type in results:
        graph_score_sum += graph_score
        for final_state, state_count in final_state_dict.items():
            if final_state in all_final_state_sum.keys():
                all_final_state_sum[final_state] += state_count
            else:
                all_final_state_sum[final_state] = state_count
        for start_state, seq_type in state_seq_type.items():
            if start_state not in all_state_seq_type.keys():
                all_state_seq_type[start_state] = {"correct": 0, "incorrect": 0, "did_not_start": 0}
            all_state_seq_type[start_state][seq_type] += 1

    return int(round(graph_score_sum / iter_count, 0)), all_final_state_sum, all_state_seq_type


def agg_count_to_csv(final_states: dict, cyclins: list, iter_count: int, filename: str):
    """
    This function will create a CSV that contains information about the final states of each of the iterations.
    """
    all_final_state_agg = dict()
    for final_state, sum_count in final_states.items():
        all_final_state_agg[final_state] = sum_count / iter_count

    data_list = [list(state) + [agg_count] for state, agg_count in all_final_state_agg.items()]

    final_state_df = pd.DataFrame(data_list, columns=cyclins + ["Avg Count"])
    csv_folder = Path("other_results", "final_state_avg")
    if not csv_folder.is_dir():
        csv_folder.mkdir(parents=True, exist_ok=True)
    csv_path = csv_folder / filename
    final_state_df.to_csv(csv_path)


def state_seq_to_csv(state_seq_count: dict, filename: str):
    """
    This function will create a CSV that contains information about the correctness of each of the iterations.
    """
    df_as_list = list()
    for start_state, state_seq in state_seq_count.items():
        df_as_list.append(
            list(start_state) + [state_seq["correct"], state_seq["incorrect"], state_seq["did_not_start"]]
        )
    df = pd.DataFrame(df_as_list, columns=[model_inputs.cyclins + ["correct", "incorrect", "did_not_start"]])
    csv_folder = Path("other_results", "state_seq")
    if not csv_folder.is_dir():
        csv_folder.mkdir(parents=True, exist_ok=True)
    csv_path = csv_folder / filename
    df.to_csv(csv_path)


def state_to_dict(nodes: list, state: str) -> dict:
    return dict(zip(nodes, list(state)))


def get_states_with_max_count(nodes: list, states_count: dict) -> tuple[int, str]:
    max_states = list()
    max_final_state_count = max(states_count.values())
    for state in [k for k, v in states_count.items() if v == max_final_state_count]:
        max_states.append(state_to_dict(nodes=nodes, state=state))
    return max_final_state_count, " | ".join(map(str, max_states))


def avg_seq_types(seq_types: dict):
    avg_seq_types = {"correct": 0, "incorrect": 0, "did_not_start": 0}

    for _, state_seq in seq_types.items():
        for seq, count in state_seq.items():
            avg_seq_types[seq] += count

    return avg_seq_types


def single_graph_execution(
    state_calc_obj: CellCycleStateCalculation,
    current_graph: list,
    graph_mod: str,
    cyclins: list,
    iterations: int,
    mp: bool = True,
):
    """
    This function calculates the score for a given graph.
    """
    state_calc_obj.set_custom_connected_graph(graph=current_graph, graph_identifier=graph_mod)

    avg_graph_score, final_state_sum, state_seq_type = score_states_multiprocess(
        state_calc_obj=state_calc_obj, iter_count=iterations, multi_process=mp
    )

    avg_seq = avg_seq_types(state_seq_type)
    max_count, max_states = get_states_with_max_count(nodes=cyclins, states_count=final_state_sum)
    print(f"For {graph_mod=}, {avg_graph_score=}")

    return graph_mod, avg_graph_score, len(final_state_sum), int(round(max_count / iterations, 0)), max_states, avg_seq


def single_perturb_details(
    state_calc_obj: CellCycleStateCalculation,
    organ: str,
    starting_graph: list,
    starting_graph_mod_id: str,
    cyclins: list,
    iter_count: int,
):
    """
    Main function for handling perturbations.
    """

    graph_image_path = draw_graph_from_matrix(organism=organ, nodes=cyclins, matrix=starting_graph)

    perturb_details = list()
    graph_mod, avg_score, unique_final_states, max_state_avg, max_state, avg_seq = single_graph_execution(
        state_calc_obj=state_calc_obj,
        current_graph=starting_graph,
        graph_mod=starting_graph_mod_id,
        cyclins=cyclins,
        iterations=iter_count,
    )
    og_graph_score = avg_score
    total_state_seq = sum(avg_seq.values())
    perturb_details.append(
        [
            graph_mod,
            round(avg_score / og_graph_score, 5),
            avg_score,
            unique_final_states,
            max_state_avg,
            max_state,
            round(100 * avg_seq["correct"] / total_state_seq, 2),
            round(100 * avg_seq["incorrect"] / total_state_seq, 2),
            round(100 * avg_seq["did_not_start"] / total_state_seq, 2),
        ]
    )

    # Cycle through perturbations and calculate data for each
    mp_args = [
        (state_calc_obj, double_perturb_graph, graph_mod_id, cyclins, iter_count)
        for double_perturb_graph, graph_mod_id in single_perturbation_generator(
            nodes=cyclins, graph=starting_graph, perturb_self_loops=True
        )
    ]
    with mp.Pool(processes=NPROC) as mp_pool:
        results = mp_pool.map(func=perturbation_mp_wrapper, iterable=mp_args)
    for graph_mod, avg_score, unique_final_states, max_state_avg, max_state, avg_seq in results:
        perturb_details.append(
            [
                graph_mod,
                round(avg_score / og_graph_score, 5),
                avg_score,
                unique_final_states,
                max_state_avg,
                max_state,
                round(100 * avg_seq["correct"] / total_state_seq, 2),
                round(100 * avg_seq["incorrect"] / total_state_seq, 2),
                round(100 * avg_seq["did_not_start"] / total_state_seq, 2),
            ]
        )

    # Write perturb data to a file
    data_file = f"{organ}_single_perturb_it{iter_count}.xlsx"
    write_perturb_data(perturb_details, graph_image_path, data_file)


def double_perturb_details(
    state_calc_obj: CellCycleStateCalculation,
    organ: str,
    starting_graph: list,
    starting_graph_mod_id: str,
    cyclins: list,
    iter_count: int,
):
    graph_image_path = draw_graph_from_matrix(organism=organ, nodes=cyclins, matrix=starting_graph)

    perturb_details = list()
    graph_mod, avg_score, unique_final_states, max_state_avg, max_state, avg_seq = single_graph_execution(
        state_calc_obj=state_calc_obj,
        current_graph=starting_graph,
        graph_mod=starting_graph_mod_id,
        cyclins=cyclins,
        iterations=iter_count,
    )
    og_graph_score = avg_score
    total_state_seq = sum(avg_seq.values())
    perturb_details.append(
        [
            graph_mod,
            round(avg_score / og_graph_score, 5),
            avg_score,
            unique_final_states,
            max_state_avg,
            max_state,
            round(100 * avg_seq["correct"] / total_state_seq, 2),
            round(100 * avg_seq["incorrect"] / total_state_seq, 2),
            round(100 * avg_seq["did_not_start"] / total_state_seq, 2),
        ]
    )

    mp_args = [
        (state_calc_obj, double_perturb_graph, graph_mod_id, cyclins, iter_count)
        for double_perturb_graph, graph_mod_id in all_perturbation_generator(
            nodes=cyclins, graph=starting_graph, perturb_self_loops=True
        )
    ]
    with mp.Pool(processes=NPROC) as mp_pool:
        results = mp_pool.map(func=perturbation_mp_wrapper, iterable=mp_args)
    for graph_mod, avg_score, unique_final_states, max_state_avg, max_state, avg_seq in results:
        perturb_details.append(
            [
                graph_mod,
                round(avg_score / og_graph_score, 5),
                avg_score,
                unique_final_states,
                max_state_avg,
                max_state,
                round(100 * avg_seq["correct"] / total_state_seq, 2),
                round(100 * avg_seq["incorrect"] / total_state_seq, 2),
                round(100 * avg_seq["did_not_start"] / total_state_seq, 2),
            ]
        )

    data_file = f"{organ}_double_perturb_it{iter_count}.xlsx"
    write_perturb_data(perturb_details, graph_image_path, data_file)


def write_perturb_data(perurbation_data: list, graph_img_path: Path, filename: str):
    """
    This function handles writing all the perturbation data to an Excel file.
    """
    df_cols = [
        "Perturbation ID",
        "Normalized Graph Score",
        "Absolute Graph Score",
        "Steady State Count",
        "Largest Attractor Size",
        "Most Frequent Steady State(s)",
        "Correct (%)",
        "Incorrect (%)",
        "Did not Start (%)",
    ]
    perturb_details_df = pd.DataFrame(perurbation_data, columns=df_cols)

    data_folder = Path("other_results", "perturbs")
    if not data_folder.is_dir():
        data_folder.mkdir(parents=True, exist_ok=True)
    file_path = data_folder / filename
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as df_writer:
        perturb_details_df.to_excel(df_writer, sheet_name="Details", index=False)
        workbook = df_writer.book
        worksheet = workbook.add_worksheet("Graph")
        worksheet.insert_image("A2", graph_img_path)
    os.remove(graph_img_path)


def write_single_graph_details(state_calc_obj: CellCycleStateCalculation, it_cnt: int, organism: str):
    """
    This function is the main function responsible for performing the simulations and gathering data on the success of each model. It will perform these calculations and place them in a csv.
    """
    avg_score, final_states_sum, state_seq_cnt = score_states_multiprocess(
        state_calc_obj=state_calc_obj, iter_count=it_cnt, multi_process=True
    )
    state_seq_to_csv(state_seq_count=state_seq_cnt, filename=f"state_seq_{it_cnt}_{organism}.csv")
    agg_count_to_csv(
        final_states=final_states_sum,
        cyclins=model_inputs.cyclins,
        iter_count=it_cnt,
        filename=f"final_state_avg_{it_cnt}_{organism}.csv",
    )
    print(f"Avg score for {organism} is {avg_score} for {it_cnt} iterations.")


def get_best_perturbation(curr_results: pd.DataFrame) -> pd.Series:
    """
    This function will return the best perturbation from the given results.
    """
    if "Exists in DB" in curr_results.columns:
        curr_results = curr_results[curr_results["Exists in DB"]]
    best_perturb = curr_results.loc[curr_results["Normalized Graph Score"].idxmin()]

    return best_perturb


def parse_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Parse arguments
    parser.add_argument(
        "--run_options",
        "-r",
        default="original",
        choices=["original", "single", "double", "perturbation"],
        nargs="+",
        help="The type of simulation to run. Defaults to execution of original graph (`original`) only. Other options are `single`, `double` and `perturbation` for single and double perturbations, and step-by-step automated perturbation analysis with single perturbations respectively.",
    )
    parser.add_argument(
        "--organism",
        "-o",
        default="model01",
        choices=["model01", "model02", "model03"],
        help="The model to use in simulation. Available models: model01, model02 and model03. Defaults to model01.",
    )
    parser.add_argument(
        "--single_iter_cnt",
        "-s",
        default=4,
        type=int,
        help="The number of iterations the simulation should run for single perturbation. Defaults to 4.",
    )
    parser.add_argument(
        "--double_iter_cnt",
        "-d",
        default=2,
        type=int,
        help="The number of iterations the simulation should run for double perturbation. Defaults to 2.",
    )
    parser.add_argument(
        "--g1_only_start_states",
        "-g1",
        action="store_true",
        help="Enables using only G1 states as start states for simulation, if the flag is present. Do not enable this if custom start states are used.",
    )
    parser.add_argument(
        "--custom_start_states",
        "-c",
        action="store_true",
        help="Enable to use custom start states as described in the input, if the flag is present.",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        default="simulation_params.yaml",
        type=str,
        help="The input file name (yaml format) to be used for the simulation. If not provided, the default input file (simulation_params.yaml) will be used.",
    )
    namespace = parser.parse_args()
    return namespace


# Main method
if __name__ == "__main__":
    start_time = time.time()

    arg_parser_obj = argparse.ArgumentParser("Cell Cycle Simulation")
    namespace = parse_arguments(arg_parser_obj)

    organism = namespace.organism
    input_models = {"model01": ModelAInputs(), "model02": ModelBInputs(), "model03": ModelCInputs()}
    model_inputs = input_models.get(organism)

    simulation_input_file = Path("sim_input", namespace.input_file)
    with open(simulation_input_file, "r") as sim_params_file:
        sim_params = yaml.safe_load(sim_params_file)

    working_graph = model_inputs.modified_graph
    cell_state_calc = CellCycleStateCalculation(
        file_inputs=sim_params, model_specific_inputs=model_inputs, user_inputs=namespace
    )

    # Enable to use filter states
    filter_states = namespace.g1_only_start_states
    # Enable to use a custom starting state
    fixed_start_states = namespace.custom_start_states

    if filter_states and fixed_start_states:
        raise ValueError("Cannot use both filter states and custom start states at the same time.")

    if filter_states:
        filtered_start_states = cell_state_calc.filter_start_states(
            one_cyclins=model_inputs.g1_state_one_cyclins, zero_cyclins=model_inputs.g1_state_zero_cyclins
        )
        cell_state_calc.set_starting_state(filtered_start_states)

    if fixed_start_states:
        cell_state_calc.set_starting_state(model_inputs.custom_start_states)

    single_it_cnt = namespace.single_iter_cnt
    double_it_cnt = namespace.double_iter_cnt

    print(
        f"Initializing '{', '.join(namespace.run_options)}' execution for {organism=}, with {filter_states=}, {fixed_start_states=}, {single_it_cnt=}, {double_it_cnt=}..."
    )

    graph_id = "Original Graph"
    # Set the graph in the primary module
    cell_state_calc.set_custom_connected_graph(graph=working_graph, graph_identifier=graph_id)

    # Extra code for debugging
    # perturb_list = list()
    # perturbations = [
    #     "Cdc25-to-CycD -> 0to1",
    #     "RB-to-CycA -> -1to0",
    #     "CycE-to-Cdc25 -> 0to-1",
    #     "CycE-to-CycA -> 0to1",
    # ]
    # for perturb in perturbations:
    #     perturb_list.append(parse_perturbation_string(perturb_str=perturb))

    # graph_id = "All-Perturbed"
    # working_graph = cell_state_calc.perturb_current_graph(perturb_list, graph_identifier=graph_id)
    # End of debugging code

    if "original" in namespace.run_options:
        write_single_graph_details(state_calc_obj=cell_state_calc, it_cnt=single_it_cnt, organism=organism)

    if "single" in namespace.run_options:
        single_perturb_details(cell_state_calc, organism, working_graph, graph_id, model_inputs.cyclins, single_it_cnt)

    if "double" in namespace.run_options:
        double_perturb_details(cell_state_calc, organism, working_graph, graph_id, model_inputs.cyclins, double_it_cnt)

    if "perturbation" in namespace.run_options:
        ...

    print(f"Inputs: {namespace}")
    end_time = time.time()
    print(
        f"Execution completed in {end_time - start_time} seconds for {single_it_cnt=}, {double_it_cnt=} for {organism} cell cycle."
    )
