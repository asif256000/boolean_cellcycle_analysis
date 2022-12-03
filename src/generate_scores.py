from copy import deepcopy

from inputs import custom_graph, cyclins, expected_final_state
from log_module import logger
from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    get_state = CellCycleStateCalculation(cyclins=cyclins, expected_final_state=expected_final_state)

    graph_scores = dict()
    for _ in range(51):
        modification_id = get_state.set_random_modified_graph(deepcopy(custom_graph))
        graph_score, final_state_dict = get_state.calculate_graph_score_and_final_states(view_state_table=False)
        graph_scores[modification_id] = graph_score
        logger.debug(f"{modification_id=}, {graph_score=}")
        # if graph_score <= 751:
        #     logger.info("Graph score less than 751...")
        #     get_state.print_final_state_count(final_state_dict)
