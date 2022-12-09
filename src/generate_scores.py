from copy import deepcopy

from inputs import custom_graph, cyclins, expected_final_state
from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    get_state = CellCycleStateCalculation(cyclins=cyclins, expected_final_state=expected_final_state)

    for _ in range(51):
        modification_id = get_state.set_random_modified_graph(deepcopy(custom_graph))
        graph_score, final_state_dict = get_state.calculate_graph_score_and_final_states(
            view_state_table=False, view_final_state_count_table=False
        )
