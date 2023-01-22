import time
from copy import deepcopy

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns


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


def multiple_mods_generator(graph: list[list]):
    for gm in single_perturbation_gen(graph=graph):
        yield from single_perturbation_gen(graph=gm)


def single_perturbation_gen(graph: list[list]):
    graph_len = len(graph)
    possible_weights = {-1, 0, 1}
    for i in range(graph_len**2):
        ix_x = i // graph_len
        ix_y = i % graph_len
        if ix_x == ix_y:
            continue
        for possible_perturb in possible_weights - {graph[ix_x][ix_y]}:
            graph_copy = deepcopy(graph)
            graph_copy[ix_x][ix_y] = possible_perturb
            yield graph_copy


def all_perturbation_generator(nodes: list, graph: list[list]):
    possible_weights = {-1, 0, 1}
    node_len = len(graph)
    for i in range(node_len**2):
        ix_x1 = i // node_len
        ix_y1 = i % node_len
        if ix_x1 == ix_y1:
            continue
        for possible_perturbs in possible_weights - {graph[ix_x1][ix_y1]}:
            cc1_graph = deepcopy(graph)
            cc1_graph[ix_x1][ix_y1] = possible_perturbs
            for j in range(i + 1, node_len**2):
                perturb_tracker_list = [
                    f"{nodes[ix_y1]}-to-{nodes[ix_x1]} -> {graph[ix_x1][ix_y1]}to{possible_perturbs}"
                ]
                ix_x2 = j // node_len
                ix_y2 = j % node_len
                if ix_x2 == ix_y2:
                    continue
                for possible_perturbs2 in possible_weights - {cc1_graph[ix_x2][ix_y2]}:
                    perturb_tracker_list.append(
                        f"{nodes[ix_y2]}-to-{nodes[ix_x2]} -> {graph[ix_x2][ix_y2]}to{possible_perturbs2}"
                    )
                    cc2_graph = deepcopy(cc1_graph)
                    cc2_graph[ix_x2][ix_y2] = possible_perturbs2
                    yield cc2_graph, " | ".join(perturb_tracker_list)
                    perturb_tracker_list.pop()


def generate_histogram(freq_list: list, img_filename: str, plot_title: str):
    sns.set_style(style="darkgrid")
    sns.histplot(freq_list, bins=75)
    plt.title(plot_title, fontsize=16)
    plt.ylabel("Frequency")
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


if __name__ == "__main__":
    nodes = ["Cln3", "MBF", "SBF", "Cln1,2", "Cdh1"]
    edges = [
        [0, -1, 1, -1, 0],
        [1, 0, 1, 0, 0],
        [0, -1, 0, 0, 0],
        [1, 0, 1, 0, -1],
        [1, -1, 0, 0, 0],
    ]
    i = 0
    for mod_graph, pert_id in all_perturbation_generator(nodes=nodes, graph=edges):
        print(f"{pert_id=}")
        for m in mod_graph:
            print(m)
        print(f"{i=}")
        i += 1

    # f_list = list(np.random.randint(low=5, high=15, size=50))
    # p1 = generate_histogram(f_list, "plot1", "First Diagram")

    # f_list = list(np.random.randint(low=5, high=15, size=50))
    # p2 = generate_histogram(f_list, "plot2", "Second Diagram")

    # fig_names = ["plot1", "plot2"]
    # combine_subplots(filename_list=fig_names)
