from inputs import cyclins, expected_final_state, modified_graph  # , custom_start_state
from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    test_state_calc = CellCycleStateCalculation(
        cyclins=cyclins, expected_final_state=expected_final_state, g1_states_only=True
    )
    # print(len(test_state_calc.start_states))
    modification_id = test_state_calc.set_custom_connected_graph(graph=modified_graph)
    # test_state_calc.set_starting_state(starting_states=[custom_start_state])
    graph_score, final_state_counts = test_state_calc.calculate_graph_score_and_final_states()
    # print(final_state_counts)  # , file=test_state_calc.output_file)
    # test_state_calc.print_final_state_count(final_state_counts)
    # print(f"{modification_id=}, {graph_score=}")  # , file=test_state_calc.output_file)
