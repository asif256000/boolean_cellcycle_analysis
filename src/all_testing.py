from state_calc import CellCycleStateCalculation
from utils import all_perturbation_generator

if __name__ == "__main__":
    organism = "yeast"
    if organism.lower() == "yeast":
        from yeast_inputs import original_graph
    else:
        from mammal_inputs import original_graph
    test_state = CellCycleStateCalculation(organism=organism)
    test_state.set_custom_connected_graph(graph=original_graph, graph_identifier="Original Graph")
    # test_state.set_starting_state(custom_start_state)
    graph_score, final_state_counts = test_state.generate_graph_score_and_final_states(
        view_state_table=False, view_final_state_count_table=True
    )
    graph_mod_tracker = 0
    for modified_matrix in all_perturbation_generator(graph=original_graph):
        test_state.set_custom_connected_graph(
            graph=modified_matrix, graph_identifier=f"Graph Modification {graph_mod_tracker:>06}"
        )
        graph_score, final_state_counts = test_state.generate_graph_score_and_final_states(
            view_state_table=False, view_final_state_count_table=True
        )
        graph_mod_tracker += 1
