import itertools
import time
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt


def _get_candidate_indices(
    graph_shape: tuple[int, int], fix_incoming_nodes: list[int], fix_outgoing_nodes: list[int], perturb_self_loops: bool
) -> list[tuple[int, int]]:
    """
    Identifies all graph indices eligible for perturbation based on a union
    of the row and column constraints.

    Args:
        graph_shape (tuple[int, int]): Shape of the graph (num_nodes, num_nodes).
        fix_incoming_nodes (list[int]): List of nodes whose incoming edges are to be perturbed.
        fix_outgoing_nodes (list[int]): List of nodes whose outgoing edges are to be perturbed.
        perturb_self_loops (bool): Whether to allow perturbation of self-loops (edges to self).

    Returns:
        list[tuple[int, int]]: List of tuples representing indices (row, col)
        that can be perturbed in the graph.
    """
    num_nodes = graph_shape[0]
    candidate_indices: set[tuple[int, int]] = set()

    # If neither list is provided, all indices are potential candidates.
    if not fix_incoming_nodes and not fix_outgoing_nodes:
        for r in range(num_nodes):
            for c in range(num_nodes):
                candidate_indices.add((r, c))
    # If any of the lists is provided, we only consider those indices.
    else:
        for r in fix_incoming_nodes:
            for c in range(num_nodes):
                candidate_indices.add((r, c))

        for c in fix_outgoing_nodes:
            for r in range(num_nodes):
                candidate_indices.add((r, c))

    # Filter out self-loops from the candidate set if required
    if not perturb_self_loops:
        candidate_indices = {(r, c) for r, c in candidate_indices if r != c}

    return sorted(list(candidate_indices))


def _generate_perturbation_chains(
    graph: npt.NDArray[np.int_], loc_combo: tuple[tuple[int, int], ...]
) -> Iterator[tuple[tuple[int, int, int], ...]]:
    """
    For a given combination of locations, generates all possible perturbation chains.
    A 'chain' is one specific combination of valid changes for each location.

    This is a generator function that yields chains one by one.

    Args:
        graph (npt.NDArray[np.int_]): The graph represented as a 2D numpy array.
        loc_combo (tuple[tuple[int, int], ...]): A tuple of tuples, where each inner tuple represents a location (row, col)
        in the graph that can be perturbed.

    Yields:
        tuple[tuple[int, int, int], ...]: A tuple of tuples, where each inner tuple represents a change at a specific location in the graph.
        Each inner tuple contains (row, col, new_value), where new_value is the value
    """
    possible_changes_per_loc = list()
    for row, col in loc_combo:
        current_value = graph[row, col]
        changes_for_this_loc = list()
        # Any value can be perturbed to the other two valid values.
        if current_value == 0:
            changes_for_this_loc.extend([(row, col, 1), (row, col, -1)])  # Can set to 1 or -1
        elif current_value == 1:
            changes_for_this_loc.extend([(row, col, 0), (row, col, -1)])  # Can set to 0 or -1
        elif current_value == -1:
            changes_for_this_loc.extend([(row, col, 0), (row, col, 1)])
        else:
            raise ValueError(f"Unexpected value {current_value} at location ({row}, {col}) in the graph.")

        if changes_for_this_loc:
            possible_changes_per_loc.append(changes_for_this_loc)

    if len(possible_changes_per_loc) == len(loc_combo):
        yield from itertools.product(*possible_changes_per_loc)


def apply_perturbation(
    graph: npt.NDArray[np.int_], perturbation: tuple[tuple[int, int, int], ...]
) -> npt.NDArray[np.int_]:
    """
    Applies a list of perturbation chains to the graph.

    Args:
        graph: A 2D NumPy array representing the directed graph's adjacency matrix.
        perturbation: A tuple of tuples where each inside tuple is (row_idx, col_idx, new_value).

    Returns:
        A new NumPy array representing the perturbed graph.
    """
    perturbed_graph = graph.copy()
    for r_idx, c_idx, new_value in perturbation:
        perturbed_graph[r_idx, c_idx] = new_value
    return perturbed_graph


def get_perturbations(
    graph: npt.NDArray[np.int_],
    perturbation_count: int = 1,
    fix_incoming_to: list[int] = list(),
    fix_outgoing_from: list[int] = list(),
    perturb_self_loops: bool = True,
) -> list[tuple[list[tuple[int, int, int]], npt.NDArray[np.int_]]]:
    """
    Generates all possible perturbed graphs by chaining a specified number of unique edge perturbations.

    Args:
        graph: A 2D NumPy array representing the directed graph's adjacency matrix.
               graph[i, j] is the edge from node j to node i.
        perturbation_count: The number of unique edges to perturb in each generated graph.
        fix_incoming_to: A list of node indices. If specified, only incoming edges
                             to these nodes (rows) will be considered for perturbation.
        fix_outgoing_to: A list of node indices. If specified, only outgoing edges
                             from these nodes (columns) will be considered for perturbation.
        perturb_self_loops: If False, self-loops (diagonal elements) will not be perturbed.

    Returns:
        A list of tuples. Each tuple contains:
        - A list of perturbation details in the format (row_idx, col_idx, new_value).
        - The resulting perturbed graph as a new NumPy array.
    """
    # --- 1. Validation ---
    if not isinstance(graph, np.ndarray) or graph.ndim != 2:
        raise ValueError("The 'graph' must be a 2D NumPy array.")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("The graph matrix must be square.")
    if perturbation_count <= 0:
        print("Warning: perturbation_count must be positive. Returning empty list.")
        return []

    # --- 2. Identify all possible locations to change ---
    candidate_indices = _get_candidate_indices(graph.shape, fix_incoming_to, fix_outgoing_from, perturb_self_loops)

    if len(candidate_indices) < perturbation_count:
        print(
            f"Warning: Found only {len(candidate_indices)} possible locations to perturb, "
            f"but requested a chain of {perturbation_count}. Returning empty list."
        )
        return []

    # --- 3. Generate all combinations of unique locations ---
    location_combinations = itertools.combinations(candidate_indices, perturbation_count)

    # --- 4. For each location combination, generate all resulting graphs ---
    final_perturbations = []
    for loc_combo in location_combinations:
        # Get all possible chained perturbations for this specific set of locations
        perturbation_chains = _generate_perturbation_chains(graph, loc_combo)

        for chain in perturbation_chains:
            # `chain` is a tuple of changes, e.g., ((r1, c1, val1), (r2, c2, val2))
            final_perturbations.append((list(chain), apply_perturbation(graph, chain)))

    return final_perturbations


def draw_graph_from_matrix(graph: npt.NDArray[np.int_], organism: str, nodes: list[str]) -> Path:
    """
    Draws a directed graph from its adjacency matrix.
    The nodes are labeled with the provided names, and edges are colored based on their weights.

    Args:
        graph: A 2D NumPy array representing the directed graph's adjacency matrix.
        organism: A string representing the name of the organism for the graph title.
        nodes: A list of strings representing the names of the nodes in the graph.

    Returns:
        pathlib.Path: The path to the saved graph image.
    """
    if not isinstance(graph, np.ndarray) or graph.ndim != 2:
        raise ValueError("The 'graph' must be a 2D NumPy array.")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("The graph matrix must be square.")
    if len(nodes) != graph.shape[0]:
        raise ValueError("The number of nodes must match the size of the graph matrix.")

    graph_image_folder = Path("figures")
    if not graph_image_folder.is_dir():
        graph_image_folder.mkdir(parents=True, exist_ok=True)
    graph_image_file = f"working_graph_{organism}_{int(time.time())}.png"
    graph_img_path = graph_image_folder / graph_image_file

    plt.figure(figsize=(10, 8))
    G = nx.MultiDiGraph()
    for node in nodes:
        G.add_node(node)

    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            curr_value = graph[i, j]
            if curr_value != 0:
                G.add_edge(nodes[j], nodes[i], weight=curr_value)

    # Separate edges by weight to draw with different arrow styles and colors
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 1]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == -1]

    pos = nx.drawing.layout.shell_layout(G, scale=2, center=(0, 0))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000, edgecolors="black")
    nx.draw_networkx_labels(G, pos)

    common_edge_attributes = {
        "width": 2,
        "connectionstyle": "arc3,rad=0.1",
        "min_target_margin": 20,
        "min_source_margin": 20,
    }
    # Draw positive edges
    if positive_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=positive_edges,
            edge_color="green",
            arrowstyle="-|>",
            arrowsize=20,
            **common_edge_attributes,
        )
    # Draw negative edges
    if negative_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=negative_edges,
            edge_color="red",
            arrowstyle="-[",
            arrowsize=10,
            **common_edge_attributes,
        )
    plt.title(f"{organism} Interaction Graph")
    plt.axis("off")
    plt.tight_layout()

    # plt.show()
    plt.savefig(graph_img_path, format="png", bbox_inches="tight", dpi=300)

    plt.close()

    return graph_img_path


# utils.py
# --- Example Usage ---
# if __name__ == "__main__":
# example_graph = np.array(
#     [
#         [0, 1, -1],
#         [1, 0, 1],
#         [1, 0, 1],
#     ],
#     dtype=np.int_,
# )

# draw_graph_from_matrix(example_graph, "Example Organism", ["A", "B", "C"])
