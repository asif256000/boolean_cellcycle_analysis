from copy import deepcopy

# from inputs import cyclins, expected_final_state, original_graph
from state_calc import CellCycleStateCalculation

# from mammal_inputs import cyclins, expected_final_state, original_graph


if __name__ == "__main__":
    organism = "yeast"
    if organism.lower() == "yeast":
        from inputs import cyclins, expected_final_state, original_graph
    else:
        from mammal_inputs import cyclins, expected_final_state, original_graph

    get_state = CellCycleStateCalculation(cyclins=cyclins, expected_final_state=expected_final_state, organism=organism)

    for _ in range(51):
        modification_id = get_state.set_random_modified_graph(deepcopy(original_graph))
        graph_score, final_state_dict = get_state.calculate_graph_score_and_final_states(
            view_state_table=True, view_final_state_count_table=True
        )
