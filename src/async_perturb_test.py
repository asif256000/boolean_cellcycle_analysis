import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd

from state_calc_clean import CellCycleStateCalculation
from utils import draw_graph_from_matrix, single_perturbation_generator

NPROC = 6


def mp_wrapper(state_calc_obj: CellCycleStateCalculation):
    (
        graph_score,
        g1_graph_score,
        final_state_dict,
        state_seq_type,
    ) = state_calc_obj.generate_graph_score_and_final_states()
    return graph_score, g1_graph_score, final_state_dict, state_seq_type


def score_states_multiprocess(iter_count: int):
    all_final_state_sum = dict()
    all_state_seq_type = dict()

    with mp.Pool(processes=NPROC) as mp_pool:
        results = mp_pool.map(
            func=mp_wrapper,
            iterable=[cell_state_calc for _ in range(iter_count)],
            chunksize=NPROC,
        )

    for graph_score, g1_graph_score, final_state_dict, state_seq_type in results:
        for final_state, state_count in final_state_dict.items():
            if final_state in all_final_state_sum.keys():
                all_final_state_sum[final_state] += state_count
            else:
                all_final_state_sum[final_state] = state_count
        for start_state, seq_type in state_seq_type.items():
            if start_state not in all_state_seq_type.keys():
                all_state_seq_type[start_state] = {"correct": 0, "incorrect": 0, "did_not_start": 0}
            all_state_seq_type[start_state][seq_type] += 1

    return all_final_state_sum, all_state_seq_type


def score_states(iter_count: int):
    all_final_state_sum = dict()
    all_state_seq_type = dict()

    for i in range(iter_count):
        (
            graph_score,
            g1_graph_score,
            final_state_dict,
            state_seq_type,
        ) = cell_state_calc.generate_graph_score_and_final_states()
        for final_state, state_count in final_state_dict.items():
            if final_state in all_final_state_sum.keys():
                all_final_state_sum[final_state] += state_count
            else:
                all_final_state_sum[final_state] = state_count
        for start_state, seq_type in state_seq_type.items():
            if start_state not in all_state_seq_type.keys():
                all_state_seq_type[start_state] = {"correct": 0, "incorrect": 0, "did_not_start": 0}
            all_state_seq_type[start_state][seq_type] += 1

    return all_final_state_sum, all_state_seq_type


def agg_count_to_csv(final_states: dict, cyclins: list, filename: str):
    all_final_state_agg = dict()
    for final_state, sum_count in final_states.items():
        all_final_state_agg[final_state] = round(sum_count / it_cnt)

    data_list = [list(state) + [agg_count] for state, agg_count in all_final_state_agg.items()]

    final_state_df = pd.DataFrame(data_list, columns=cyclins + ["Avg Count"])
    csv_path = Path("other_results", filename)
    final_state_df.to_csv(csv_path)


def state_seq_to_csv(state_seq_count: dict, filename: str):
    df_as_list = list()
    for start_state, state_seq in state_seq_count.items():
        df_as_list.append(
            list(start_state) + [state_seq["correct"], state_seq["incorrect"], state_seq["did_not_start"]]
        )
    df = pd.DataFrame(df_as_list, columns=[cyclins + ["correct", "incorrect", "did_not_start"]])
    df.to_csv(Path("other_results", filename))


def state_to_dict(nodes: list, state: str) -> dict:
    return dict(zip(nodes, list(state)))


def get_states_with_max_count(nodes: list, states_count: dict) -> dict:
    max_states = list()
    max_final_state_count = max(states_count.values())
    for state in [k for k, v in states_count.items() if v == max_final_state_count]:
        max_states.append(state_to_dict(nodes=nodes, state=state))
    return max_final_state_count, " | ".join(map(str, max_states))


def execute_perturb_mp():
    with mp.Pool(processes=NPROC) as mp_pool:
        results = mp_pool.map(
            func=mp_wrapper,
            iterable=[
                cell_state_calc for _ in range(single_perturbation_generator(nodes=cyclins, graph=working_graph))
            ],
            chunksize=NPROC,
        )


def single_perturb_details():
    (
        graph_score,
        g1_graph_score,
        final_state_dict,
        state_seq_type,
    ) = cell_state_calc.generate_graph_score_and_final_states()
    graph_image_path = Path("figures", "working_graph.png")
    draw_graph_from_matrix(nodes=cyclins, matrix=working_graph, graph_img_path=graph_image_path)

    max_count, max_states = get_states_with_max_count(nodes=cyclins, states_count=final_state_dict)
    perturb_details = [[state_seq_type, len(final_state_dict), max_count, max_states]]

    for single_perturb_graph, graph_mod_id in single_perturbation_generator(nodes=cyclins, graph=working_graph):
        (
            graph_score,
            g1_graph_score,
            final_state_dict,
            state_seq_type,
        ) = cell_state_calc.generate_graph_score_and_final_states(graph_info=(single_perturb_graph, graph_mod_id))
        max_count, max_states = get_states_with_max_count(nodes=cyclins, states_count=final_state_dict)
        perturb_details.append([state_seq_type, len(final_state_dict), max_count, max_states])

    perturb_details_df = pd.DataFrame(
        perturb_details, columns=["Graph Modification ID", "Unique State Count", "Max State Count", "Max State(s)"]
    )
    data_path = Path("other_results", f"{organism}_perturb.xlsx")

    with pd.ExcelWriter(data_path, engine="xlsxwriter") as df_writer:
        perturb_details_df.to_excel(df_writer, sheet_name="Details", index=False)
        workbook = df_writer.book
        worksheet = workbook.add_worksheet("Graph")
        worksheet.insert_image("A2", graph_image_path)
    os.remove(graph_image_path)


if __name__ == "__main__":
    organism = "fr_mammal"

    if organism.lower() == "yeast":
        from yeast_inputs import cyclins, g1_state_one_cyclins, g1_state_zero_cyclins, modified_graph, original_graph

        target_ix = 7
    elif organism.lower() == "gb_mammal":
        from gb_mammal_inputs import (
            cyclins,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
            modified_graph,
            original_graph,
        )

        target_ix = 1
    elif organism.lower() == "fr_mammal":
        from fr_mammal_inputs import (
            cyclins,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
            modified_graph,
            original_graph,
        )

        target_ix = 1

    calc_params = {
        "cyclins": cyclins,
        "organism": organism,
        "detailed_logs": True,
        "hardcoded_self_loops": True,
        "check_sequence": True,
        "g1_states_only": False,
        "view_state_table": True,
        "view_state_changes_only": True,
        "view_final_state_count_table": True,
        "async_update": True,
        "random_order_cyclin": True,
        "complete_cycle": False,
        "expensive_state_cycle_detection": True,
        "cell_cycle_activation_cyclin": cyclins[target_ix],
    }

    filter_states = True

    working_graph = modified_graph
    cell_state_calc = CellCycleStateCalculation(input_json=calc_params)
    cell_state_calc.set_custom_connected_graph(graph=working_graph, graph_identifier="OG Graph")

    if filter_states:
        filtered_start_states = cell_state_calc.filter_start_states(
            one_cyclins=g1_state_one_cyclins, zero_cyclins=g1_state_zero_cyclins
        )
        cell_state_calc.set_starting_state(filtered_start_states)

    # single_perturb_details()

    it_cnt = 1000
    final_states_sum, state_seq_cnt = score_states_multiprocess(iter_count=it_cnt)
    # final_states_sum, state_seq_count = score_states(iter_count=it_cnt)

    state_seq_to_csv(state_seq_count=state_seq_cnt, filename=f"state_seq_{it_cnt}_{organism}.csv")
    agg_count_to_csv(
        final_states=final_states_sum, cyclins=cyclins, filename=f"final_state_avg_{it_cnt}_{organism}.csv"
    )
