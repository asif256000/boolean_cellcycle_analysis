from inputs import custom_graph, cyclins, expected_final_state  # , custom_start_state
from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    test_state_calc = CellCycleStateCalculation(cyclins=cyclins, expected_final_state=expected_final_state)
    modification_id = test_state_calc.set_custom_connected_graph(graph=custom_graph)
    # test_state_calc.set_starting_state(starting_states=[custom_start_state])
    graph_score, final_state_counts = test_state_calc.calculate_graph_score_and_final_states(
        view_state_table=False, view_final_state_count_table=False
    )
