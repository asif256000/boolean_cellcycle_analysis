from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    organism = "yeast"
    if organism.lower() == "yeast":
        from yeast_inputs import original_graph
    else:
        from mammal_inputs import original_graph

    test_state = CellCycleStateCalculation(organism=organism)
    modification_id = test_state.set_custom_connected_graph(graph=original_graph, graph_identifier="Original Graph")
    graph_score, final_state_counts = test_state.generate_graph_score_and_final_states(
        view_state_table=True, view_final_state_count_table=True
    )
