import multiprocessing as mp
import os
from pathlib import Path
from time import time

import pandas as pd

from state_calc_clean import CellCycleStateCalculation
from utils import all_perturbation_generator, draw_graph_from_matrix, single_perturbation_generator

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
    graph_score_sum = 0
    all_final_state_sum = dict()
    all_state_seq_type = dict()

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
    df_as_list = list()
    for start_state, state_seq in state_seq_count.items():
        df_as_list.append(
            list(start_state) + [state_seq["correct"], state_seq["incorrect"], state_seq["did_not_start"]]
        )
    df = pd.DataFrame(df_as_list, columns=[cyclins + ["correct", "incorrect", "did_not_start"]])
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


def write_perturb_data(perurbation_data: list, df_cols: list, graph_img_path: Path, file_path: Path):
    perturb_details_df = pd.DataFrame(perurbation_data, columns=df_cols)

    with pd.ExcelWriter(file_path, engine="xlsxwriter") as df_writer:
        perturb_details_df.to_excel(df_writer, sheet_name="Details", index=False)
        workbook = df_writer.book
        worksheet = workbook.add_worksheet("Graph")
        worksheet.insert_image("A2", graph_img_path)
    os.remove(graph_img_path)


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
    graph_image_folder = Path("figures")
    if not graph_image_folder.is_dir():
        graph_image_folder.mkdir(parents=True, exist_ok=True)
    graph_image_file = f"working_graph_{organ}_{int(time())}.png"
    graph_image_path = graph_image_folder / graph_image_file
    draw_graph_from_matrix(nodes=cyclins, matrix=starting_graph, graph_img_path=graph_image_path)

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

    data_folder = Path("other_results", "perturbs")
    if not data_folder.is_dir():
        data_folder.mkdir(parents=True, exist_ok=True)
    data_file = f"{organ}_single_perturb_it{iter_count}.xlsx"
    data_path = data_folder / data_file
    data_cols = [
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
    write_perturb_data(perturb_details, data_cols, graph_image_path, data_path)


def double_perturb_details(
    state_calc_obj: CellCycleStateCalculation,
    organ: str,
    starting_graph: list,
    starting_graph_mod_id: str,
    cyclins: list,
    iter_count: int,
):
    graph_image_folder = Path("figures")
    if not graph_image_folder.is_dir():
        graph_image_folder.mkdir(parents=True, exist_ok=True)
    graph_image_file = f"working_graph_{organ}_{int(time())}.png"
    graph_image_path = graph_image_folder / graph_image_file
    draw_graph_from_matrix(nodes=cyclins, matrix=starting_graph, graph_img_path=graph_image_path)

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

    data_folder = Path("other_results", "perturbs")
    if not data_folder.is_dir():
        data_folder.mkdir(parents=True, exist_ok=True)
    data_file = f"{organ}_double_perturb_it{iter_count}.xlsx"
    data_path = data_folder / data_file
    data_cols = [
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
    write_perturb_data(perturb_details, data_cols, graph_image_path, data_path)


def write_single_graph_details(state_calc_obj: CellCycleStateCalculation, it_cnt: int):
    avg_score, final_states_sum, state_seq_cnt = score_states_multiprocess(
        state_calc_obj=state_calc_obj, iter_count=it_cnt, multi_process=True
    )
    state_seq_to_csv(state_seq_count=state_seq_cnt, filename=f"state_seq_{it_cnt}_{organism}.csv")
    agg_count_to_csv(
        final_states=final_states_sum,
        cyclins=cyclins,
        iter_count=it_cnt,
        filename=f"final_state_avg_{it_cnt}_{organism}.csv",
    )
    print(f"Avg score for {organism} is {avg_score} for {it_cnt} iterations.")


if __name__ == "__main__":
    start_time = time()
    organism = "model01"

    if organism.lower() == "model01":
        from model01_inputs import (
            custom_start_states,
            cyclins,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
            modified_graph,
        )

        target_ix = 7
    elif organism.lower() == "model02":
        from model02_inputs import (
            custom_start_states,
            cyclins,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
            modified_graph,
        )

        target_ix = 1
    elif organism.lower() == "model03":
        from model03_inputs import (
            custom_start_states,
            cyclins,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
            modified_graph,
        )

        target_ix = 1

    calc_params = {
        "cyclins": cyclins,  # Input according to the organism passed
        "organism": organism,  # Organism is passed as a string. Possible values: "model01", "model02", "model03"
        "detailed_logs": False,  # Generates large log files with detailed execution information
        "hardcoded_self_loops": True,  # Hardcoded self loops for the graph. If this is True, self-loops in the input is followed, otherwise self-loops are calculated at runtime according to incoming and outgoing edges
        "check_sequence": True,  # Checks if the state sequence follows the expected order or not
        "g1_states_only": False,  # Start states are fixed to G1 states only if this is True (G1 states are defined in the input file)
        "view_state_table": False,  # Prints the state table in the log file for each iteration
        "view_state_changes_only": True,  # Prints only the state changes and states where there is no change is ignored in the log file
        "view_final_state_count_table": True,  # Prints the final state count table in the log file
        "async_update": True,  # If True, the state of the nodes are updated asynchronously, otherwise the state of the nodes are updated synchronously
        "random_order_cyclin": True,  # If True, the cyclins are updated in a random order, otherwise the cyclins are updated in the order of the cyclins list
        "complete_cycle": False,  # If True, the order of cyclin update, even when randomly picked, always picks all the nodes in the cyclin before picking the same cyclin again
        "expensive_state_cycle_detection": True,  # Recommended to keep True. If True, the state cycle detection is done by checking for cycle of any length in the end of state sequence, otherwise the state cycle detection is done by comparing the last two states only
        "cell_cycle_activation_cyclin": cyclins[
            target_ix
        ],  # The cyclin that is used to detect the cell cycle activation. If this is never activated, the cell cycle is categorized as 'Did not Start'
        "max_updates_per_cycle": 150,  # Number of updates in every state cycle. The bigger this number is, the more likely that the final state reaches a steady state, but it also takes more time to compute
    }

    filter_states = False

    working_graph = modified_graph
    cell_state_calc = CellCycleStateCalculation(input_json=calc_params)

    if filter_states:
        filtered_start_states = cell_state_calc.filter_start_states(
            one_cyclins=g1_state_one_cyclins, zero_cyclins=g1_state_zero_cyclins
        )
        cell_state_calc.set_starting_state(filtered_start_states)

    fixed_start_states = False

    if fixed_start_states:
        cell_state_calc.set_starting_state(custom_start_states)

    single_it_cnt = 500
    double_it_cnt = 20

    print(
        f"Initializing execution for {organism=}, with {filter_states=}, {fixed_start_states=}, {single_it_cnt=}, {double_it_cnt=}..."
    )

    cell_state_calc.set_custom_connected_graph(graph=working_graph, graph_identifier="Original Graph")

    write_single_graph_details(state_calc_obj=cell_state_calc, it_cnt=single_it_cnt)
    # write_single_graph_details(state_calc_obj=cell_state_calc, it_cnt=double_it_cnt)

    single_perturb_details(cell_state_calc, organism, working_graph, "Original Graph", cyclins, single_it_cnt)
    double_perturb_details(cell_state_calc, organism, working_graph, "Original Graph", cyclins, double_it_cnt)

    end_time = time()
    print(
        f"Execution completed in {end_time - start_time} seconds for {single_it_cnt=}, {double_it_cnt=} for {organism} cell cycle."
    )
