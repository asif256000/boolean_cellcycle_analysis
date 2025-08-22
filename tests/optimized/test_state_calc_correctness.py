
import numpy as np
import pytest
from argparse import Namespace
import jax.numpy as jnp

from src.all_inputs import ModelAInputs
from src.state_calc_clean import CellCycleStateCalculation as OriginalStateCalc
from optimized.state_calc import CellCycleStateCalculator as OptimizedStateCalc

@pytest.fixture
def setup_calculators():
    """Sets up both the original and optimized calculators with the same inputs."""
    file_inputs = {
        "check_sequence": False,
        "async_update": False,
        "g1_states_only": False,
        "max_updates_per_cycle": 10,
        "detailed_logs": False,
        "hardcoded_self_loops": True,
        "view_state_table": False,
        "view_state_changes_only": False,
        "view_final_state_count_table": False,
        "random_order_cyclin": False,
        "complete_cycle": False,
        "expensive_state_cycle_detection": False,
    }
    user_inputs = Namespace(organism="model01")
    model_inputs = ModelAInputs()

    original_calc = OriginalStateCalc(file_inputs, model_inputs, user_inputs)
    optimized_calc = OptimizedStateCalc(file_inputs, model_inputs, user_inputs)

    original_calc.set_custom_connected_graph(model_inputs.modified_graph)
    optimized_calc._CellCycleStateCalculator__model_graph = np.array(model_inputs.modified_graph, dtype=np.int_)

    return original_calc, optimized_calc

def test_initialization_consistency(setup_calculators):
    """Tests that both calculators initialize with the same state."""
    original_calc, optimized_calc = setup_calculators

    # Compare relevant attributes
    assert original_calc._CellCycleStateCalculation__all_cyclins == optimized_calc._CellCycleStateCalculator__all_cyclins
    assert np.array_equal(np.array(original_calc._CellCycleStateCalculation__start_states), optimized_calc._CellCycleStateCalculator__start_states)
    assert original_calc._CellCycleStateCalculation__organism == optimized_calc._CellCycleStateCalculator__organism

def test_filter_starting_states_consistency(setup_calculators):
    """Tests that both calculators filter starting states consistently."""
    original_calc, optimized_calc = setup_calculators

    zero_cyclins = ["Cln3", "MBF"]
    one_cyclins = ["SBF"]

    original_filtered_states = original_calc.filter_start_states(zero_cyclins, one_cyclins)
    optimized_calc.filter_starting_states(zero_cyclins, one_cyclins)

    assert np.array_equal(np.array(original_filtered_states, dtype=np.int_), optimized_calc._CellCycleStateCalculator__start_states)

def test_sync_calculate_next_state_consistency(setup_calculators):
    """Tests that both calculators produce the same next state in synchronous mode."""
    original_calc, optimized_calc = setup_calculators

    original_states = np.array(original_calc._CellCycleStateCalculation__start_states, dtype=np.int_)
    optimized_states = optimized_calc._CellCycleStateCalculator__start_states

    original_next_states = original_calc._CellCycleStateCalculation__sync_calculate_next_step(
        original_calc.nodes_and_edges, original_states[0].tolist()
    )
    optimized_next_states = optimized_calc._sync_calculate_next_state(
        jnp.array(optimized_calc._CellCycleStateCalculator__model_graph), jnp.array([optimized_states[0]])
    )

    assert np.array_equal(np.array(original_next_states), np.array(optimized_next_states).flatten())

def test_async_calculate_next_state_consistency(setup_calculators):
    """Tests that both calculators produce the same next state in asynchronous mode."""
    original_calc, optimized_calc = setup_calculators

    original_states = np.array(original_calc._CellCycleStateCalculation__start_states, dtype=np.int_)
    optimized_states = optimized_calc._CellCycleStateCalculator__start_states

    for i in range(len(original_calc._CellCycleStateCalculation__all_cyclins)):
        original_next_states = original_calc._CellCycleStateCalculation__async_calculate_next_step(
            original_calc.nodes_and_edges, original_states[0].tolist(), i
        )
        optimized_next_states = optimized_calc._async_calculate_next_state(
            jnp.array(optimized_calc._CellCycleStateCalculator__model_graph), jnp.array([optimized_states[0]]), cyclin_ix=i
        )

        assert np.array_equal(np.array(original_next_states), np.array(optimized_next_states).flatten())
