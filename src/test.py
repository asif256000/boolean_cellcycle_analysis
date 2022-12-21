# from inputs import cyclins, expected_final_state, original_graph  # modified_graph

# from mammal_inputs import cyclins, expected_final_state, original_graph  # modified_graph
from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    organism = "mammal"
    if organism.lower() == "yeast":
        from inputs import cyclins, expected_final_state, original_graph
    else:
        from mammal_inputs import cyclins, expected_final_state, original_graph

    test_state = CellCycleStateCalculation(
        cyclins=cyclins, expected_final_state=expected_final_state, organism=organism
    )
    modification_id = test_state.set_custom_connected_graph(graph=original_graph)
    # test_state_calc.set_starting_state(starting_states=[custom_start_state])
    graph_score, final_state_counts = test_state.calculate_graph_score_and_final_states(
        view_state_table=True, view_final_state_count_table=True
    )
