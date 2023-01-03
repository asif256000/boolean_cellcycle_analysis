from copy import deepcopy

graph = {
    "NODE01": {
        0: {"NODE02", "NODE03", "NODE04"},
        1: set(),
        -1: {"NODE05"},
    },
    "NODE02": {
        0: {"NODE04", "NODE05"},
        1: {"NODE03"},
        -1: {"NODE01"},
    },
    "NODE03": {
        0: {"NODE02", "NODE05"},
        1: {"NODE04"},
        -1: {"NODE01"},
    },
    "NODE04": {
        0: {"NODE01", "NODE05"},
        1: {"NODE02"},
        -1: {"NODE03"},
    },
    "NODE05": {
        0: {"NODE03"},
        1: {"NODE01", "NODE02", "NODE04"},
        -1: set(),
    },
}


def all_2_pertub_generator(graph: dict[str, dict], start_pos_marker: tuple = (None, None, None)) -> dict:
    """Create all possible two-edge perturbations for a graph. Generator is used to return the result so that large number of graphs are not stored in memory.

    :param dict graph: Original graph for which perturbations are to be calculated.
    :return dict: Graph with two perturbations to the original graph.
    :yield Iterator[dict]: Generator of the perturbed graphs.
    """
    all_possible_weights = {-1, 0, 1}
    target_node = start_pos_marker[0]
    curr_edge_weight = start_pos_marker[1]
    origin_node = start_pos_marker[2]

    for node, edge_map in graph.items():
        while node != target_node:
            continue
        for edge_weight, origin_nodes in edge_map.items():
            while edge_weight != curr_edge_weight:
                continue
            for o_node in origin_nodes:
                while o_node != origin_node:
                    continue
                l1_cc_graph = deepcopy(graph)
                l1_cc_graph[node][edge_weight].difference_update({o_node})
                to_weight = all_possible_weights.difference({edge_weight})
                l1_cc_graph[node][to_weight].update({o_node})
                yield all_2_pertub_generator(l1_cc_graph, start_pos_marker=(node, to_weight, o_node))


if __name__ == "__main__":
    for mod_graph in all_2_pertub_generator(graph):
        print(mod_graph)
