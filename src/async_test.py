import multiprocessing as mp
from pathlib import Path

import pandas as pd

from state_calc_clean import CellCycleStateCalculation

NPROC = 6


def mp_wrapper(state_calc_obj: CellCycleStateCalculation):
    (
        graph_score,
        g1_graph_score,
        final_state_dict,
        graph_mod_id,
    ) = state_calc_obj.generate_graph_score_and_final_states()
    return graph_score, g1_graph_score, final_state_dict, graph_mod_id


def score_states_multiprocess(iter_count: int):
    all_final_state_sum = dict()
    with mp.Pool(processes=NPROC) as mp_pool:
        results = mp_pool.map(
            func=mp_wrapper,
            iterable=[cell_state_calc for _ in range(iter_count)],
            chunksize=NPROC,
        )
    for graph_score, g1_graph_score, final_state_dict, graph_mod_id in results:
        for final_state, state_count in final_state_dict.items():
            if final_state in all_final_state_sum.keys():
                all_final_state_sum[final_state] += state_count
            else:
                all_final_state_sum[final_state] = state_count

    return all_final_state_sum


def score_states(iter_count: int):
    all_final_state_sum = dict()

    for i in range(iter_count):
        (
            graph_score,
            g1_graph_score,
            final_state_dict,
            graph_mod_id,
        ) = cell_state_calc.generate_graph_score_and_final_states()
        for final_state, state_count in final_state_dict.items():
            if final_state in all_final_state_sum.keys():
                all_final_state_sum[final_state] += state_count
            else:
                all_final_state_sum[final_state] = state_count

    return all_final_state_sum


def agg_count_to_csv(final_state_agg, cyclins, filename):
    data_list = [list(state) + [agg_count] for state, agg_count in final_state_agg.items()]

    final_state_df = pd.DataFrame(data_list, columns=cyclins + ["Avg Count"])
    csv_path = Path("other_results", filename)
    final_state_df.to_csv(csv_path)


if __name__ == "__main__":
    organism = "fr_mammal"

    if organism.lower() == "yeast":
        from yeast_inputs import cyclins, modified_graph, original_graph
    elif organism.lower() == "gb_mammal":
        from gb_mammal_inputs import (
            cyclins,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
            modified_graph,
            original_graph,
        )
    elif organism.lower() == "fr_mammal":
        from fr_mammal_inputs import (
            cyclins,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
            modified_graph,
            original_graph,
        )

    calc_params = {
        "cyclins": cyclins,
        "organism": organism,
        "detailed_logs": True,
        "hardcoded_self_loops": True,
        "g1_states_only": False,
        "view_state_table": True,
        "view_state_changes_only": False,
        "view_final_state_count_table": True,
        "async_update": True,
        "random_order_cyclin": True,
        "complete_cycle": False,
        "expensive_state_cycle_detection": True,
    }

    all_final_state_agg = dict()

    cell_state_calc = CellCycleStateCalculation(input_json=calc_params)
    cell_state_calc.set_custom_connected_graph(graph=modified_graph, graph_identifier="Mod Graph")
    filtered_start_states = cell_state_calc.filter_start_states(
        one_cyclins=g1_state_one_cyclins, zero_cyclins=g1_state_zero_cyclins
    )
    cell_state_calc.set_starting_state(filtered_start_states)

    it_cnt = 1

    # final_states_sum = score_states_multiprocess(iter_count=it_cnt)
    final_states_sum = score_states(iter_count=it_cnt)

    for final_state, sum_count in final_states_sum.items():
        all_final_state_agg[final_state] = round(sum_count / it_cnt)

    # agg_count_to_csv(
    #     final_state_agg=all_final_state_agg, cyclins=cyclins, filename=f"final_state_avg_{it_cnt}_{organism}.csv"
    # )
