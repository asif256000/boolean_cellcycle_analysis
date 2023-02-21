from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    organism = "yeast"
    if organism.lower() == "yeast":
        from yeast_inputs import modified_graph, original_graph
    else:
        from gb_mammal_inputs import modified_graph, original_graph

    get_state = CellCycleStateCalculation(organism=organism)
    get_state = CellCycleStateCalculation(organism=organism)

    modification_id = get_state.set_custom_connected_graph(graph=modified_graph, graph_identifier="Modified Graph")
    graph_score, g1_graph_score, final_state_dict = get_state.generate_graph_score_and_final_states(
        view_state_table=True, view_final_state_count_table=True
    )
