import time
from copy import deepcopy
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from input_testing import cyclins, modified_graph, original_graph


def all_perturbation_recursive_generator(graph: list[list], start_pos: int = 0):
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
            x = all_perturbation_recursive_generator(graph=cc1_graph, start_pos=i + 1)
            yield x
        if ix_x1 == node_len - 1 and ix_y1 == node_len - 1:
            raise StopIteration


def all_perturbation_generator(nodes: list, graph: list[list], perturb_self_loops: bool = False):
    possible_weights = {-1, 0, 1}
    node_len = len(graph)
    for i in range(node_len**2 + 1):
        ix_x1 = i // node_len
        ix_y1 = i % node_len
        if not perturb_self_loops and ix_x1 == ix_y1:
            continue
        for possible_perturbs1 in possible_weights - {graph[ix_x1][ix_y1]}:
            cc1_graph = deepcopy(graph)
            cc1_graph[ix_x1][ix_y1] = possible_perturbs1
            for j in range(i + 1, node_len**2):
                perturb_tracker_list = [
                    f"{nodes[ix_y1]}-to-{nodes[ix_x1]} -> {graph[ix_x1][ix_y1]}to{possible_perturbs1}"
                ]
                ix_x2 = j // node_len
                ix_y2 = j % node_len
                if not perturb_self_loops and ix_x2 == ix_y2:
                    continue
                for possible_perturbs2 in possible_weights - {cc1_graph[ix_x2][ix_y2]}:
                    perturb_tracker_list.append(
                        f"{nodes[ix_y2]}-to-{nodes[ix_x2]} -> {graph[ix_x2][ix_y2]}to{possible_perturbs2}"
                    )
                    cc2_graph = deepcopy(cc1_graph)
                    cc2_graph[ix_x2][ix_y2] = possible_perturbs2
                    yield cc2_graph, " | ".join(perturb_tracker_list)
                    perturb_tracker_list.pop()


def single_perturbation_generator(nodes: list, graph: list[list], perturb_self_loops: bool = False):
    possible_weights = {-1, 0, 1}
    node_len = len(graph)
    for i in range(node_len**2):
        ix_x = i // node_len
        ix_y = i % node_len
        if not perturb_self_loops:
            if ix_x == ix_y:
                continue
        for possible_perturbs in possible_weights - {graph[ix_x][ix_y]}:
            cc_graph = deepcopy(graph)
            cc_graph[ix_x][ix_y] = possible_perturbs
            yield cc_graph, f"{nodes[ix_y]}-to-{nodes[ix_x]} -> {graph[ix_x][ix_y]}to{possible_perturbs}"


def generate_histogram(freq_list: list, img_filename: str, plot_title: str, vertical_line_at: int = None):
    sns.set_style(style="white")
    plt.figure(figsize=(80, 40))
    ax = sns.histplot(freq_list, bins=100, color="black")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title("", fontsize=16)
    plt.ylabel("")
    if vertical_line_at:
        plt.axvline(vertical_line_at, color="red", linestyle="--")
    plt.savefig(img_filename, bbox_inches="tight")
    plt.close()


def generate_categorical_hist(
    freq_dict: dict[str, int], img_filename: str, plot_title: str, vertical_line_at: int = None
):
    sns.set_style(style="white")
    wid = len(freq_dict) // 2.5
    plt.figure(figsize=(wid, 20))
    sns.barplot(x=list(freq_dict.keys()), y=list(freq_dict.values()))
    # plt.title(plot_title)
    # plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    if vertical_line_at:
        plt.axvline(x=vertical_line_at, color="red", linestyle="--")
    plt.savefig(img_filename, bbox_inches="tight")
    plt.close()


def combine_subplots(filename_list: list, plot_title: str = "Frequency Charts"):
    fig, axis = plt.subplots(len(filename_list))
    fig.suptitle(plot_title, fontsize=20)
    for ix, ax in enumerate(axis.flat):
        ax.set_axis_off()
        ax.imshow(mpimg.imread(filename_list[ix] + ".png"))
    plt.tight_layout()
    fig_name = f"fig_{time.strftime('%m%d_%H%M%S', time.gmtime(time.time()))}"
    plt.savefig(fig_name)
    plt.close()


def add_graph_edges(all_states: list, existing_edges: list):
    existing_edges.append(("start_state", all_states[0]))
    for ix in range(len(all_states) - 1):
        existing_edges.append((all_states[ix], all_states[ix + 1]))
    existing_edges.append((all_states[-1], "end_state"))


def draw_complete_graph(all_graph_edges: list, fig_size: tuple = (50, 50), graph_img_name: str = "test_graph"):
    G = nx.MultiDiGraph()
    plt.figure(figsize=fig_size)
    G.add_edges_from(list(set(all_graph_edges)))
    # pos = nx.drawing.layout.circular_layout(G)
    nx.draw_networkx(
        G,
        with_labels=True,
        node_shape="o",
        label="NewGraph",
        nodelist=["start", "end"],
    )
    plt.savefig(graph_img_name)
    plt.close()


def get_all_edges(all_paths: list[list]) -> list:
    graph_edges = list()
    for single_path in all_paths:
        graph_edges.append([single_path[0], single_path[1]])
        for ix, node in enumerate(graph_edges[1:-2]):
            graph_edges.append([node, single_path[ix + 1]])
    return graph_edges


def draw_graph_from_matrix(nodes: list, matrix: list[list], graph_img_path: Path = Path("interaction_graph")):
    G = nx.MultiDiGraph()

    color_map = {1: "green", -1: "red"}
    for in_cyc_ix, incoming_edges in enumerate(matrix):
        for out_cyc_ix, out_edge in enumerate(incoming_edges):
            if out_edge != 0:
                G.add_edge(
                    nodes[out_cyc_ix],
                    nodes[in_cyc_ix],
                    weight=out_edge,
                    color=color_map[out_edge],
                    width=1.5,
                    label=out_edge,
                )

    edge_color_list = list(nx.get_edge_attributes(G, "color").values())
    width_list = list(nx.get_edge_attributes(G, "width").values())
    # pos = nx.drawing.layout.circular_layout(G)
    pos = nx.drawing.layout.shell_layout(G)
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_shape="",
        edge_color=edge_color_list,
        width=width_list,
        connectionstyle="arc3, rad = 0.1",
        arrowstyle="-|>",
        # label=graph_img_path.stem,
    )
    # plt.show()
    plt.title(graph_img_path.stem)
    plt.savefig(graph_img_path, bbox_inches="tight")
    plt.close()


def flatten_double_perturb_freq(scores: dict[str, int]):
    total_scores = dict()
    counts = dict()

    for pert, score in scores.items():
        perturbations = pert.split(" | ")
        for perturb in perturbations:
            if perturb in total_scores.keys():
                total_scores[perturb] += score
            else:
                total_scores[perturb] = score
            if perturb in counts.keys():
                counts[perturb] += 1
            else:
                counts[perturb] = 1

    # print(f"{counts=}")
    return {pert: total_scores[pert] / counts[pert] for pert in total_scores.keys()}


if __name__ == "__main__":
    data = pd.read_excel(
        "/Users/asifiqbal/code_projects/DNABool/other_results/perturbs/gb_mammal_double_perturb_it8.xlsx",
        sheet_name="Details",
    )

    # For Score Data
    # scores = data["Graph Score"]
    # # threshold = 40000
    # # mod_data = np.where(scores > threshold, threshold + 1, scores)
    # original_graph_score = data[data["Graph Modification ID"] == "OG Graph"]["Graph Score"].values[0]
    # generate_histogram(
    #     freq_list=scores,
    #     img_filename="Yeast_noticks.png",
    #     plot_title="Yeast Single Perturbation Score Distribution (512 iterations)",
    #     vertical_line_at=original_graph_score,
    # )

    # For Perturbation Data
    perturb_data = data.set_index("Graph Modification ID")["Graph Score"].to_dict()
    # filtered_data = {key: value for key, value in perturb_data.items() if value < perturb_data["OG Graph"] + 25}
    flat_data = flatten_double_perturb_freq(perturb_data)
    generate_categorical_hist(
        freq_dict=flat_data,
        img_filename="GB_Mammal_Perturbation.png",
        plot_title="Double Perturbation Async Scores",
        vertical_line_at=list(flat_data.keys()).index("OG Graph"),
    )

    # nodes = ["Cln3", "MBF", "SBF", "Cln1,2", "Cdh1"]
    # edges = [
    #     [0, -1, 1, -1, 0],
    #     [1, 0, 1, 0, 0],
    #     [0, -1, 1, 0, 0],
    #     [1, 0, 1, 0, -1],
    #     [1, -1, 0, 0, 0],
    # ]
    # i = 0
    # for pert_graph, graph_mod in single_perturbation_generator(nodes=nodes, graph=edges):
    #     print(f"{pert_graph=}, {graph_mod=}")
    #     i += 1
    # print(i)
    # draw_graph_from_matrix(cyclins, original_graph, graph_img_path=Path("figures", "original_graph.png"))
    # draw_graph_from_matrix(cyclins, modified_graph, graph_img_path=Path("figures", "modified_graph.png"))
    # for mod_graph, pert_id in all_perturbation_generator(nodes=nodes, graph=edges):
    #     print(f"{pert_id=}")
    #     for m in mod_graph:
    #         print(m)
    #     print(f"{i=}")
    #     i += 1

    # f_list = list(np.random.randint(low=5, high=15, size=50))
    # p1 = generate_histogram(f_list, "plot1", "First Diagram")

    # f_list = list(np.random.randint(low=5, high=15, size=50))
    # p2 = generate_histogram(f_list, "plot2", "Second Diagram")

    # fig_names = ["plot1", "plot2"]
    # combine_subplots(filename_list=fig_names)
