from state_calc import CellCycleStateCalculation

if __name__ == "__main__":
    organism = "mammal"
    if organism.lower() == "yeast":
        from yeast_inputs import cyclins, modified_graph, original_graph
    else:
        from gb_mammal_inputs import cyclins, modified_graph, original_graph

    get_state = CellCycleStateCalculation(
        cyclins=cyclins, organism=organism, detailed_logs=True, async_update=True, random_order_cyclin=False
    )

    modification_id = get_state.set_custom_connected_graph(graph=original_graph, graph_identifier="OG Graph")
    graph_score, g1_graph_score, final_state_dict = get_state.generate_graph_score_and_final_states(
        view_state_table=True, view_final_state_count_table=True
    )
