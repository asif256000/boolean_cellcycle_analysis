import pytest
import numpy as np
from argparse import Namespace
from src.all_inputs import ModelAInputs
from src.state_calc_clean import CellCycleStateCalculation
from optimized.state_calc import CellCycleStateCalculator

# Mock file_inputs for testing
mock_file_inputs = {
    "detailed_logs": False,
    "hardcoded_self_loops": False,
    "check_sequence": True,
    "g1_states_only": False,
    "view_state_table": False,
    "view_state_changes_only": False,
    "view_final_state_count_table": False,
    "async_update": True,
    "random_order_cyclin": True,
    "complete_cycle": False,
    "expensive_state_cycle_detection": False,
    "max_updates_per_cycle": 50,
    "max_iterations": 50, # Added for optimized version
    "force_cpu_only": True,
}

# Mock user_inputs for testing
mock_user_inputs = Namespace(organism="model01")

@pytest.fixture
def common_inputs():
    model_inputs = ModelAInputs()
    return mock_file_inputs, model_inputs, mock_user_inputs

def test_generate_graph_score_and_final_states_consistency(common_inputs):
    file_inputs, model_inputs, user_inputs = common_inputs

    original_calculator = CellCycleStateCalculation(file_inputs, model_inputs, user_inputs)
    optimized_calculator = CellCycleStateCalculator(file_inputs, model_inputs, user_inputs)

    # Use the same graph matrix for both calculators
    graph_matrix = model_inputs.modified_graph
    original_calculator.set_custom_connected_graph(graph_matrix, graph_identifier="Original Graph")

    # Run simulation for original calculator
    original_graph_score, original_final_state_count, original_state_seq_types = \
        original_calculator.generate_graph_score_and_final_states()

    # Run simulation for optimized calculator
    # The optimized calculator's generate_graph_score_and_final_states expects graph_matrix and graph_mod_id
    optimized_graph_score, optimized_final_state_count, optimized_state_seq_types = \
        optimized_calculator.generate_graph_score_and_final_states(
            graph_matrix=np.array(graph_matrix, dtype=np.int_),
            graph_mod_id="Original Graph"
        )

    # Compare graph scores
    assert np.isclose(original_graph_score, optimized_graph_score), "Graph scores do not match"

    # Compare final state counts
    assert original_final_state_count == optimized_final_state_count, "Final state counts do not match"

    # Compare state sequence types
    assert original_state_seq_types == optimized_state_seq_types, "State sequence types do not match"
