from copy import deepcopy

from inputs import custom_graph, cyclins, expected_final_state
from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    get_state = CellCycleStateCalculation(cyclins=cyclins, expected_final_state=expected_final_state)

    graph_scores = dict()
    for _ in range(50):
        modification_id = get_state.set_random_modified_graph(deepcopy(custom_graph))
        # modification_id = get_state.set_custom_connected_graph(graph=custom_graph)
        # get_state.set_starting_state(starting_states=[custom_start_state])
        graph_score = get_state.calculate_graph_score()
        graph_scores[modification_id] = graph_score
        print(f"{modification_id=}, {graph_score=}")

    print(f"{graph_scores}")
