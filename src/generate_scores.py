import time
from copy import deepcopy

from inputs import custom_graph, cyclins, expected_final_state
from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    get_state = CellCycleStateCalculation(cyclins=cyclins, expected_final_state=expected_final_state)

    graph_scores = dict()
    for _ in range(51):
        modification_id = get_state.set_random_modified_graph(deepcopy(custom_graph))
        # modification_id = get_state.set_custom_connected_graph(graph=custom_graph)
        # get_state.set_starting_state(starting_states=[custom_start_state])
        graph_score = get_state.calculate_graph_score()
        graph_scores[modification_id] = graph_score
        if graph_score <= 751:
            print(f"{modification_id=}, {graph_score=}")

        # print(f"{graph_scores}")

    with open(f"results/{time.strftime('%m%d_%H%M%S', time.gmtime(time.time()))}.txt", "w") as score_file:
        score_file.writelines([f"{k}: {v}\n" for k, v in graph_scores.items()])
