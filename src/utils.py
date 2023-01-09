from copy import deepcopy


def all_perturbation_recursive_generation(graph: list[list], start_pos: int = 0, iter_count: int = 0):
    possible_weights = {-1, 0, 1}
    node_len = len(graph)
    for i in range(start_pos, node_len**2):
        ix_x1 = i // node_len
        ix_y1 = i % node_len
        if ix_x1 == ix_y1:
            continue
        for possible_pertubs in possible_weights - {graph[ix_x1][ix_y1]}:
            cc1_graph = deepcopy(graph)
            cc1_graph[ix_x1][ix_y1] = possible_pertubs
            yield all_perturbation_recursive_generation(graph=cc1_graph, start_pos=i + 1, iter_count=iter_count + 1)


def all_perturbation_generator(graph: list[list]):
    possible_weights = {-1, 0, 1}
    node_len = len(graph)
    for i in range(node_len**2):
        ix_x1 = i // node_len
        ix_y1 = i % node_len
        if ix_x1 == ix_y1:
            continue
        for possible_pertubs in possible_weights - {graph[ix_x1][ix_y1]}:
            cc1_graph = deepcopy(graph)
            cc1_graph[ix_x1][ix_y1] = possible_pertubs
            for j in range(i + 1, node_len**2):
                ix_x2 = j // node_len
                ix_y2 = j % node_len
                if ix_x2 == ix_y2:
                    continue
                for possible_perturbs2 in possible_weights - {cc1_graph[ix_x2][ix_y2]}:
                    cc2_graph = deepcopy(cc1_graph)
                    cc2_graph[ix_x2][ix_y2] = possible_perturbs2
                    yield cc2_graph


if __name__ == "__main__":
    nodes = ["1", "2", "3", "4", "5"]
    edges = [
        [0, -1, 1, -1, 0],
        [1, 0, 1, 0, 0],
        [0, -1, 0, 0, 0],
        [1, 0, 1, 0, -1],
        [1, -1, 0, 0, 0],
    ]
    i = 1
    # for matrix in all_perturbation_generation(edges):
    for matrix in all_perturbation_generator(edges):
        for lst in matrix:
            print(lst)
        print(i)
        i += 1
