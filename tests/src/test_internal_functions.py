from argparse import Namespace

import numpy as np
import pytest

from src.state_calc_clean import CellCycleStateCalculation


@pytest.fixture
def mock_inputs():
    """Fixture to create a mock instance of CellCycleStateCalculation with necessary attributes."""

    class MockModelSpecificInputs:
        cyclins = ["NodeA", "NodeB", "NodeC"]
        expected_final_state = [1, 0, 1]
        cell_cycle_activation_cyclin = "NodeA"
        optimal_graph_score = 10
        g1_only_optimal_graph_score = 5
        rule_based_self_activation = False
        rule_based_self_deactivation = False
        g1_state_zero_cyclins = []
        g1_state_one_cyclins = ["NodeA"]
        expected_final_state = [1, "-", 1]
        expected_cyclin_order = [{"NodeA": 1, "NodeB": 0}, {"NodeB": 1}]
        modified_graph = [[0, 1, -1], [1, 0, 1], [1, -1, 1]]

    file_inputs = {
        "detailed_logs": False,
        "hardcoded_self_loops": True,
        "check_sequence": True,
        "g1_states_only": False,
        "view_state_table": False,
        "view_state_changes_only": False,
        "view_final_state_count_table": False,
        "async_update": True,
        "random_order_cyclin": True,
        "complete_cycle": False,
        "expensive_state_cycle_detection": True,
        "max_updates_per_cycle": 100,
    }

    user_inputs = Namespace(organism="TestOrganism")

    cell_cycle_object = CellCycleStateCalculation(file_inputs, MockModelSpecificInputs(), user_inputs)
    cell_cycle_object.set_custom_connected_graph(
        graph=MockModelSpecificInputs.modified_graph, graph_identifier="TestGraph"
    )
    return cell_cycle_object


@pytest.mark.parametrize(
    "current_state, cyclin_ix, expected_state",
    [
        ([1, 1, 0], 0, [1, 1, 0]),
        ([1, 0, 1], 1, [1, 1, 1]),
        ([0, 1, 1], 2, [0, 1, 0]),
        ([1, 0, 1], 2, [1, 0, 1]),
    ],
)
def test_async_calculate_next_step(mock_inputs, current_state, cyclin_ix, expected_state):
    """Test the __async_calculate_next_step function."""
    next_state = mock_inputs._CellCycleStateCalculation__async_calculate_next_step(
        mock_inputs.nodes_and_edges, current_state, cyclin_ix
    )
    assert next_state == expected_state, f"Expected {expected_state}, got {next_state}"


@pytest.mark.parametrize(
    "current_state, expected_next",
    [
        ([0, 0, 1], [0, 1, 1]),
        ([1, 0, 0], [1, 1, 1]),
        ([0, 1, 1], [0, 1, 0]),
    ],
)
def test_sync_calculate_next_step(mock_inputs, current_state, expected_next):
    """Test the __sync_calculate_next_step function."""
    next_state = mock_inputs._CellCycleStateCalculation__sync_calculate_next_step(
        mock_inputs.nodes_and_edges, current_state
    )
    assert next_state == expected_next, f"Expected {expected_next}, got {next_state}"


# @pytest.mark.parametrize(
#     "current_state, next_state, cyclin_ix, expected_next",
#     [
#         ([1, 1, 0], [1, 1, 0], 2, [1, 1, 1]),
#         ([0, 0, 1], [0, 0, 1], 0, [0, 0, 0]),
#     ],
# )
# def test_decide_self_loops(mock_inputs, current_state, next_state, cyclin_ix, expected_next):
#     """Test the __decide_self_loops function."""
#     mock_inputs._CellCycleStateCalculation__decide_self_loops(
#         graph_matrix=mock_inputs.nodes_and_edges,
#         current_state=current_state,
#         next_state=next_state,
#         cyclin_ix=cyclin_ix,
#     )
#     assert next_state == expected_next, f"Expected {expected_next}, got {next_state}"


@pytest.mark.parametrize(
    "final_state, expected_score",
    [
        ([1, 0, 1], 0),
        ([1, 0, 1], 0),
        ([0, 1, 0], 2),
        ([1, 1, 0], 1),
    ],
)
def test_calculate_state_scores(mock_inputs, final_state, expected_score):
    """Test the __calculate_state_scores method."""
    score = mock_inputs._CellCycleStateCalculation__calculate_state_scores(final_state)
    assert score == expected_score, f"Expected {expected_score}, got {score}"


# @pytest.mark.parametrize(
#     "all_states, expected_result",
#     [
#         ([[1, 0, 1], [1, 0, 1]], False),
#         ([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]], True),
#         ([[1, 1, 1], [1, 1, 1], [1, 1, 0]], False),
#     ],
# )
# def test_detect_end_cycles(mock_inputs, all_states, expected_result):
#     """Test the __detect_end_cycles method."""
#     cycle_detected = mock_inputs._CellCycleStateCalculation__detect_end_cycles(all_states)
#     assert cycle_detected == expected_result, f"Expected {expected_result}, got {cycle_detected}"
