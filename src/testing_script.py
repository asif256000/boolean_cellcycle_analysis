import multiprocessing as mp
import random
import time
import uuid

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from utils import add_graph_edges, draw_complete_graph, generate_categorical_hist


def test(x: int, y: int):
    return x * y, x + y


def generator_test(dummy_list: list, some_list: list[tuple]):
    for x, y in some_list:
        print(f"Sleeping for {(x+y)//2} seconds...")
        time.sleep((x + y) // 2)
        yield x + y, abs(x - y)
    time.sleep(0.2)
    print(f"Generator Complete. {dummy_list=}")


def some_test_fn(test_tup: tuple[int, int], dummy1: int = 0, dummy2: str = "null"):
    sum, diff = test_tup[0], test_tup[1]
    print(f"Sum: {sum}, Difference: {diff}")
    return sum, diff


def mp_test(some_tup: tuple[int, int]):
    print(f"First: {some_tup[0]}, Second: {some_tup[1]}")
    return some_tup[0], some_tup[1]


def queue_mp_handler(q: mp.Queue):
    args = q.get()
    return test(args[0], args[1])


def mp_handler(args: tuple):
    return test(args[0], args[1])


def modify_matrix(matrix, shuffle_list):
    n = len(matrix)
    if len(shuffle_list) != n:
        raise ValueError("The length of the shuffle_list must be equal to the number of rows in the matrix.")
    new_matrix = list(np.zeros_like(matrix))
    for i, j in enumerate(shuffle_list):
        new_matrix[i] = [matrix[j][x] for x in shuffle_list]
    return new_matrix


def detect_cycle(arr: list) -> bool:
    """Should return True iff there is an ongoing cycle at the end of the list. Cycle here can be defined as a repeated sequence
    that appears in the list in the same order multiple times consecutively.

    :param list arr: The list in which cycle has to be detected.
    :return bool: Returns True iff a cycle is detected at the end of the list, False otherwise.
    """
    # return any(arr[-i:] == arr[-2 * i : -i] for i in range(2, len(arr) // 2 + 1))
    arr_len = len(arr)
    rev_arr = list(reversed(arr))
    for rev_ix, elem in enumerate(rev_arr[: arr_len // 2 + 1]):
        for i in range(2, arr_len // 2):
            if rev_arr[rev_ix : rev_ix + i] == rev_arr[rev_ix + i : rev_ix + 2 * i]:
                return True
    return False


def draw_all_paths(nx_obj: nx.MultiDiGraph, all_path: list = list()):
    edges_list = [("start_state", all_path[0])]
    for ix in range(len(all_path) - 1):
        edges_list.append((all_path[ix], all_path[ix + 1]))
    edges_list.append((all_path[-1], "end_state"))

    nx_obj.add_edges_from(edges_list)

    # return G
    # plt.figure(figsize=(8, 8))
    # nx.draw(G)
    # plt.savefig("graph_name")


if __name__ == "__main__":
    all_edges = list()
    sample_states = [f"{i:>05b}" for i in range(2**5)]
    for i in range(20):
        path_list = random.choices(population=sample_states, k=9)
        add_graph_edges(all_states=path_list, existing_edges=all_edges)

    print(f"{all_edges=} with length={len(all_edges)}, unique count={len(list(set(all_edges)))}")

    draw_complete_graph(all_graph_edges=all_edges, graph_img_name="test")
    # edges = [(1, 2), (2, 1), (3, 2), (2, 4), ("start_state", "end_state")]
    # draw_complete_graph(edges, "test")
    # state_path01 = ["10100", "11101", "11111", "10000", "10100", "10011", "01110", "11110", "00111"]
    # draw_all_paths(G, state_path01)
    # state_path02 = ["01001", "10100", "00010", "11110", "01010", "11110", "01110", "00001", "10100", "01100", "11001"]
    # draw_all_paths(G, state_path02)
    # state_path03 = ["10101", "10101", "10110", "00100", "00101", "01100", "10111", "00000", "10111", "10011"]
    # plt.figure(figsize=(16, 16))
    # nx.draw(G, with_labels=True, node_shape="o", label="NewGraph", nodelist=["start_state", "end_state"])
    # plt.savefig("graph_name")

    # cyc_arr = [
    #     "1101",
    #     "1001",
    #     "1010",
    #     "1010",
    #     "1010",
    #     "0011",
    #     "0011",
    # "1111",
    # "1110",
    # "0111",
    # "1101",
    # "1010",
    # "0101",
    # "1101",
    # "1010",
    # "0101",
    # "1101",
    #     "1010",
    #     "0101",
    #     "1101",
    #     "1010",
    #     "0011",
    #     "0011",
    # ]
    # prev = object()
    # filtered_list = [prev := v for v in cyc_arr if prev != v]
    # filtered_list = [v for i, v in enumerate(cyc_arr) if i == 0 or v != cyc_arr[i - 1]]
    # print(filtered_list)
    # res = detect_cycle(cyc_arr)
    # print(f"{res=} for {filtered_list=}")
