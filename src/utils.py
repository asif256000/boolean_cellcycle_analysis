import re
import time
from copy import deepcopy
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

perturbation_format_string = "{src_node}-to-{dest_node} -> {old_weight}to{new_weight}"


def parse_perturbation_string(perturb_str: str):
    perturb = perturb_str.split(" -> ")
    nodes, weights = perturb[0], perturb[-1]
    src_node, dest_node = nodes.split("-to-")
    old_weight, new_weight = weights.split("to")

    result = {
        "src_node": src_node,
        "dest_node": dest_node,
        "old_weight": old_weight,
        "new_weight": new_weight,
    }
    return result


def parse_perturbation_string_with_regex(perturb_str: str):
    perturbation_re_pattern = re.compile(
        r"(?P<src_node>.+?)-to-(?P<dest_node>.+?) -> (?P<old_weight>.+?)to(?P<new_weight>.+?)"
    )
    match = perturbation_re_pattern.match(perturb_str)
    return match.groupdict()


def _apply_perturbation(nodes, graph, dest_ix, src_ix, old_wt, new_wt):
    cc_graph = deepcopy(graph)
    cc_graph[dest_ix][src_ix] = new_wt
    source, destination = nodes[src_ix], nodes[dest_ix]
    return cc_graph, perturbation_format_string.format(
        src_node=source, dest_node=destination, old_weight=old_wt, new_weight=new_wt
    )


def __get_possible_perturbs_for_position(graph: list[list], row_ix: int, col_ix: int) -> list[tuple]:
    """
    Get possible perturbations for a given position in the graph.

    :param list[list] graph: The graph represented as a 2D list.
    :param int row_ix: The row index of the position.
    :param int col_ix: The column index of the position.
    :return list[tuple]: A list of tuples representing the possible perturbations. The tuple contains the source index, destination index, old weight, and new weights.
    """
    possible_weights = {-1, 0, 1}
    old_wt = graph[row_ix][col_ix]
    return [(col_ix, row_ix, old_wt, new_wt) for new_wt in possible_weights - {old_wt}]


def __get_specific_node_single_perturbs(
    nodes: list, graph: list[list], nodes_to_perturb: list[str] = None, perturb_self_loops: bool = False
) -> list[tuple]:
    all_possible_perturbations = list()
    node_len = len(graph)
    if nodes_to_perturb is None:
        nodes_to_perturb = nodes

    for node in nodes_to_perturb:
        node_ix = nodes.index(node)
        for i in range(node_len):
            if not perturb_self_loops and i == node_ix:
                continue
            all_possible_perturbations.extend(__get_possible_perturbs_for_position(graph, node_ix, i))
            if i == node_ix:
                continue
            all_possible_perturbations.extend(__get_possible_perturbs_for_position(graph, i, node_ix))

    return all_possible_perturbations


def specific_node_perturbation_generator(
    nodes: list, graph: list[list], nodes_to_perturb: list[str] = None, perturb_self_loops: bool = False
):
    all_single_perturbations = __get_specific_node_single_perturbs(nodes, graph, nodes_to_perturb, perturb_self_loops)

    for src_ix, dest_ix, old_wt, new_wt in all_single_perturbations:
        yield _apply_perturbation(nodes, graph, dest_ix, src_ix, old_wt, new_wt)


def specific_node_double_perturbation_generator(
    nodes: list, graph: list[list], nodes_to_perturb: list[str] = None, perturb_self_loops: bool = False
):
    track_perturbations = set()
    all_single_perturbations = __get_specific_node_single_perturbs(nodes, graph, nodes_to_perturb, perturb_self_loops)
    for perturb1 in all_single_perturbations:
        for perturb2 in all_single_perturbations:
            if perturb1 == perturb2:
                continue
            src_ix1, dest_ix1, old_wt1, new_wt1 = perturb1
            src_ix2, dest_ix2, old_wt2, new_wt2 = perturb2
            if src_ix1 == src_ix2 and dest_ix1 == dest_ix2:
                continue

            pair_key = frozenset([perturb1, perturb2])
            if pair_key in track_perturbations:
                continue
            track_perturbations.add(pair_key)

            graph1, perturb_str1 = _apply_perturbation(nodes, graph, dest_ix1, src_ix1, old_wt1, new_wt1)
            graph2, perturb_str2 = _apply_perturbation(nodes, graph1, dest_ix2, src_ix2, old_wt2, new_wt2)

            yield graph2, f"{perturb_str1} | {perturb_str2}"


def specific_node_multi_perturbation_generator(
    nodes: list,
    graph: list[list],
    nodes_to_perturb: list = None,
    perturbation_depth: int = 2,
    perturb_self_loops: bool = False,
):
    all_single_perturbations = __get_specific_node_single_perturbs(nodes, graph, nodes_to_perturb, perturb_self_loops)
    seen_combinations = set()

    def _recursive_apply(base_graph, depth, chosen, used, used_positions):
        if depth == perturbation_depth:
            key = frozenset(chosen)
            if key not in seen_combinations:
                seen_combinations.add(key)
                yield base_graph, " | ".join(chosen)
            return
        for p in all_single_perturbations:
            if p in used:
                continue
            src_ix, dest_ix, old_wt, new_wt = p
            position_key = (src_ix, dest_ix)
            if position_key in used_positions:
                continue
            used_copy = set(used)
            used_copy.add(p)
            positions_copy = set(used_positions)
            positions_copy.add(position_key)

            updated_graph, p_str = _apply_perturbation(nodes, base_graph, dest_ix, src_ix, old_wt, new_wt)
            yield from _recursive_apply(updated_graph, depth + 1, chosen + [p_str], used_copy, positions_copy)

    yield from _recursive_apply(graph, 0, [], set(), set())


def double_perturbation_generator(nodes: list, graph: list[list], perturb_self_loops: bool = False):
    possible_weights = {-1, 0, 1}
    node_len = len(graph)
    for i in range(node_len**2):
        ix_x1 = i // node_len
        ix_y1 = i % node_len
        if not perturb_self_loops and ix_x1 == ix_y1:
            continue

        for possible_perturbs1 in possible_weights - {graph[ix_x1][ix_y1]}:
            cc1_graph = deepcopy(graph)
            cc1_graph[ix_x1][ix_y1] = possible_perturbs1
            for j in range(i + 1, node_len**2):
                source, destination = nodes[ix_y1], nodes[ix_x1]
                old_wt, new_wt = graph[ix_x1][ix_y1], possible_perturbs1
                perturb_tracker_list = [
                    perturbation_format_string.format(
                        src_node=source, dest_node=destination, old_weight=old_wt, new_weight=new_wt
                    )
                ]
                ix_x2 = j // node_len
                ix_y2 = j % node_len
                if not perturb_self_loops and ix_x2 == ix_y2:
                    continue
                for possible_perturbs2 in possible_weights - {cc1_graph[ix_x2][ix_y2]}:
                    sec_source, sec_destination = nodes[ix_y2], nodes[ix_x2]
                    sec_old_wt, sec_new_wt = cc1_graph[ix_x2][ix_y2], possible_perturbs2
                    perturb_tracker_list.append(
                        perturbation_format_string.format(
                            src_node=sec_source, dest_node=sec_destination, old_weight=sec_old_wt, new_weight=sec_new_wt
                        )
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
            source, destination = nodes[ix_y], nodes[ix_x]
            old_wt, new_wt = graph[ix_x][ix_y], possible_perturbs
            yield cc_graph, perturbation_format_string.format(
                src_node=source, dest_node=destination, old_weight=old_wt, new_weight=new_wt
            )


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
    sns.barplot(x=list(freq_dict.keys()), y=list(freq_dict.values()), color="black")
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


def draw_graph_from_matrix(organism: str, nodes: list, matrix: list[list]):
    graph_image_folder = Path("figures")
    if not graph_image_folder.is_dir():
        graph_image_folder.mkdir(parents=True, exist_ok=True)
    graph_image_file = f"working_graph_{organism}_{int(time.time())}.png"
    graph_img_path = graph_image_folder / graph_image_file

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

    return graph_img_path


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
        "/Users/asifiqbal/code_projects/DNABool/other_results/perturbs/gb_mammal_single_perturb_it256.xlsx",
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
    filtered_data = {key: value for key, value in perturb_data.items() if value < perturb_data["OG Graph"] + 5}
    # flat_data = flatten_double_perturb_freq(perturb_data)
    generate_categorical_hist(
        freq_dict=filtered_data,
        img_filename="GB_Mammal_Perturbation.png",
        plot_title="Single Perturbation Async Scores",
        vertical_line_at=list(filtered_data.keys()).index("OG Graph"),
    )
