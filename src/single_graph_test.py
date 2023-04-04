from state_calc import CellCycleStateCalculation

# from state_calc_clean import CellCycleStateCalculation

if __name__ == "__main__":
    organism = "mammal"
    if organism.lower() == "yeast":
        from yeast_inputs import cyclins, modified_graph, original_graph
    else:
        from gb_mammal_inputs import cyclins, modified_graph, original_graph

    calc_params = {
        "cyclins": cyclins,
        "organism": "goldbeter_mammal",
        "detailed_logs": True,
        "g1_states_only": False,
        "async_update": True,
        "random_order_cyclin": True,
        "complete_cycle": True,
    }

    # get_state = CellCycleStateCalculation(input_json=calc_params)
    get_state = CellCycleStateCalculation(
        cyclins=calc_params["cyclins"],
        organism=calc_params["organism"],
    )

    get_state.set_custom_connected_graph(graph=original_graph, graph_identifier="OG Graph")
    graph_score, g1_graph_score, final_state_dict = get_state.generate_graph_score_and_final_states(
        view_state_table=True, view_final_state_count_table=True
    )

    # Try loop with 100s of iterations and note paths, scores etc
    # list_of_scores = list()
