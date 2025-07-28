import itertools

import numpy as np
import pytest
from scipy.special import comb

# Assuming the function is in `optimized/utils.py` and tests are run from the root directory
from optimized.utils import get_perturbations


@pytest.fixture
def graph_3x3():
    """A standard 3x3 graph for most test cases."""
    return np.array([[0, 1, -1], [1, 0, 0], [1, 0, 1]], dtype=np.int_)


@pytest.fixture
def graph_5x5():
    """A larger 5x5 graph for scalability checks, filled with zeros."""
    return np.zeros((5, 5), dtype=np.int_)


def test_zero_perturbation_count(graph_3x3):
    """Test that a perturbation_count of 0 returns an empty list."""
    assert get_perturbations(graph_3x3, 0) == []


@pytest.mark.parametrize("graph_fixture_name", ["graph_3x3", "graph_5x5"])
def test_single_perturbations(graph_fixture_name, request):
    """Test the generation of single perturbations on multiple graph sizes."""
    graph = request.getfixturevalue(graph_fixture_name)
    perturbations = get_perturbations(graph, 1)

    # Each cell can be perturbed in 2 ways.
    expected_count = graph.size * 2
    assert len(perturbations) == expected_count

    # A generic check to ensure the first returned perturbation is valid.
    if perturbations:
        details, p_graph = perturbations[0]
        r, c, new_val = details[0]

        # The value at the perturbed location should be the new value.
        assert p_graph[r, c] == new_val
        # The original graph's value at that location should be different.
        assert graph[r, c] != new_val
        # The number of differing elements between the graphs should be exactly 1.
        assert np.sum(graph != p_graph) == 1


@pytest.mark.parametrize("graph_fixture_name", ["graph_3x3", "graph_5x5"])
def test_chained_perturbations(graph_fixture_name, request):
    """Test chained perturbations of length 2 on multiple graph sizes."""
    graph = request.getfixturevalue(graph_fixture_name)
    perturbations = get_perturbations(graph, 2)

    # C(N, 2) combinations of locations, each has 2*2=4 outcomes.
    expected_count = comb(graph.size, 2, exact=True) * 4
    assert len(perturbations) == expected_count

    # A generic check for validity of a chained perturbation.
    if perturbations:
        details, p_graph = perturbations[0]
        assert len(details) == 2
        # The number of differing elements between the graphs should be exactly 2.
        assert np.sum(graph != p_graph) == 2


@pytest.mark.parametrize(
    "graph_fixture_name, kwargs, expected_count, description",
    [
        # --- Cases for 3x3 graph ---
        ("graph_3x3", {"fix_incoming_to": [0]}, 6, "3x3: Fix incoming to row 0 -> 3 cells * 2 changes = 6"),
        ("graph_3x3", {"fix_outgoing_from": [1]}, 6, "3x3: Fix outgoing from col 1 -> 3 cells * 2 changes = 6"),
        (
            "graph_3x3",
            {"fix_incoming_to": [0], "fix_outgoing_from": [0]},
            10,
            "3x3: Union of row 0/col 0 -> 5 cells * 2 changes = 10",
        ),
        ("graph_3x3", {"perturb_self_loops": False}, 12, "3x3: No self-loops -> 6 cells * 2 changes = 12"),
        (
            "graph_3x3",
            {"fix_incoming_to": [0], "perturb_self_loops": False},
            4,
            "3x3: Row 0, no self-loops -> 2 cells * 2 changes = 4",
        ),
        (
            "graph_3x3",
            {"perturbation_count": 6, "fix_incoming_to": [0], "fix_outgoing_from": [0]},
            0,
            "3x3: Perturbation count (6) > candidates (5) -> 0",
        ),
        # --- Cases for 5x5 graph ---
        ("graph_5x5", {"perturbation_count": 1}, 50, "5x5: Single perturbation -> 25 cells * 2 changes = 50"),
        (
            "graph_5x5",
            {"perturbation_count": 2, "fix_incoming_to": [0]},
            40,
            "5x5: Chained (2) in row 0 -> C(5,2)*4 = 40",
        ),
        (
            "graph_5x5",
            {"perturbation_count": 1, "perturb_self_loops": False},
            40,
            "5x5: Single, no self-loops -> 20 cells * 2 changes = 40",
        ),
        (
            "graph_5x5",
            {"fix_incoming_to": [0, 1], "fix_outgoing_from": [0]},
            26,
            "5x5: Union of rows 0,1 & col 0 -> 13 cells * 2 changes = 26",
        ),
    ],
)
def test_all_constraints(graph_fixture_name, kwargs, expected_count, description, request):
    """
    Tests various constraints against multiple graph sizes.
    It uses pytest's 'request' fixture to dynamically fetch the graph fixture by name.
    """
    # Dynamically get the graph fixture based on the parameterized name
    graph = request.getfixturevalue(graph_fixture_name)

    perturbations = get_perturbations(graph, **kwargs)
    assert len(perturbations) == expected_count, f"Failed on case: {description}"


@pytest.mark.parametrize(
    "graph, exc, match_msg",
    [
        (np.array([[0, 1, 2], [3, 4, 5]]), ValueError, "The graph matrix must be square."),
        (np.array([1, 2, 3]), ValueError, "The 'graph' must be a 2D NumPy array."),
        ([[0, 1], [1, 0]], ValueError, "The 'graph' must be a 2D NumPy array."),
    ],
)
def test_input_validation(graph, exc, match_msg):
    """Test that the function raises ValueErrors for invalid input."""
    with pytest.raises(exc, match=match_msg):
        get_perturbations(graph, 1)


def test_perturbation_details_are_correct(graph_3x3):
    """Verify that generated details correctly reflect the change in the graph."""
    perturbations = get_perturbations(graph_3x3, 1)

    details, perturbed_graph = perturbations[0]
    r, c, new_val = details[0]

    assert perturbed_graph[r, c] == new_val
    original_copy = graph_3x3.copy()
    original_copy[r, c] = new_val
    assert np.array_equal(perturbed_graph, original_copy)
