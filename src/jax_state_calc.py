"""
A modified version of the state calculator with JAX acceleration.

This file integrates the JAX accelerator module into the original state calculator
to enable GPU-accelerated batch processing of states. Key features include:

1. Automatic GPU/CPU detection and selection
2. Dynamic batch size calculation based on model complexity
3. Seamless fallback to original implementation when JAX cannot be used
4. Memory optimization for large models
5. Support for both synchronous and asynchronous update modes
"""

import logging
import time
from argparse import Namespace

import numpy as np

# Try to import JAX accelerator and memory optimizer
try:
    from src.jax_accelerator import (
        JAX_AVAILABLE,
        calculate_optimal_batch_size,
        configure_jax_devices,
        is_jax_compatible,
        process_states_in_batches,
    )

    # Import memory optimization if available
    try:
        from src.utils.jax_memory_optimizer import (
            adaptive_batch_size,
            estimate_memory_requirements,
            memory_efficient_state_processing,
        )

        MEMORY_OPTIMIZER_AVAILABLE = True
        logging.info("JAX memory optimizer loaded successfully")
    except ImportError:
        MEMORY_OPTIMIZER_AVAILABLE = False
        logging.warning("JAX memory optimizer not available")

    logging.info("JAX acceleration module loaded successfully")
except ImportError:
    JAX_AVAILABLE = False
    MEMORY_OPTIMIZER_AVAILABLE = False
    logging.warning("JAX acceleration module not available")

from src.state_calc import CellCycleStateCalculation


class JaxCellCycleCalculation(CellCycleStateCalculation):
    """
    JAX-accelerated version of the CellCycleStateCalculation class.

    This class extends the original implementation with JAX-based optimizations
    for faster state calculations, particularly for large state spaces.
    """

    def __init__(self, file_inputs: dict, model_specific_inputs, user_inputs: Namespace) -> None:
        """Initialize with JAX acceleration options"""
        # Add JAX optimization parameters
        self.__use_gpu = file_inputs.get("use_gpu", True)
        self.__use_cpu = file_inputs.get("use_cpu", True)
        self.__force_cpu_only = file_inputs.get("force_cpu_only", False)
        self.__batch_size_gpu = file_inputs.get("batch_size_gpu", None)

        # Call parent initialization
        super().__init__(file_inputs, model_specific_inputs, user_inputs)

        # Configure JAX if available
        if JAX_AVAILABLE:
            self.gpu_devices, self.cpu_devices = configure_jax_devices(
                self.__use_gpu, self.__use_cpu, self.__force_cpu_only
            )

            if self.gpu_devices and not self.__force_cpu_only:
                self.primary_device = self.gpu_devices[0]
            elif self.cpu_devices:
                self.primary_device = self.cpu_devices[0]
            else:
                self.primary_device = None

            self.batch_size_gpu = calculate_optimal_batch_size(
                len(self._CellCycleStateCalculation__all_cyclins),
                self.__batch_size_gpu,
                self.gpu_devices,
                self.__force_cpu_only,
            )

            logging.debug(
                f"JAX configuration: GPU devices={len(self.gpu_devices)}, "
                f"CPU devices={len(self.cpu_devices)}, "
                f"Batch size={self.batch_size_gpu}"
            )
        else:
            self.gpu_devices = []
            self.cpu_devices = []
            self.primary_device = None
            self.batch_size_gpu = 64
            logging.warning("JAX acceleration is not available")

    def __iterate_all_start_states(self, graph_matrix, graph_mod_id):
        """
        This method iterates through all possible start states of the cell cycle model and
        simulates the cell cycle dynamics with JAX acceleration when possible.

        Args:
            graph_matrix: The graph matrix representing the network
            graph_mod_id: Identifier for the graph modification

        Returns:
            tuple: (state_scores_dict, final_states, state_seq_type)
        """
        # Check if we can use JAX acceleration
        can_use_jax = is_jax_compatible(
            async_update=self._CellCycleStateCalculation__async_update,
            self_activation=self._CellCycleStateCalculation__self_activation_flag,
            self_deactivation=self._CellCycleStateCalculation__self_deactivation_flag,
            force_cpu_only=self.__force_cpu_only,
        )

        if can_use_jax:
            logging.debug("Using JAX-optimized state processing")

            # Select states based on configuration
            if self._CellCycleStateCalculation__g1_states_only_flag:
                states_to_process = self._CellCycleStateCalculation__g1_start_states
            else:
                states_to_process = self._CellCycleStateCalculation__start_states

            # Process using JAX acceleration
            t_start = time.time()
            try:
                # Process all states in batches
                batch_results = process_states_in_batches(
                    states_to_process,
                    graph_matrix,
                    batch_size=self.batch_size_gpu,
                    max_iterations=self._CellCycleStateCalculation__max_iter_count,
                    device=self.primary_device,
                )

                # Organize results in the expected format
                state_scores_dict = {}
                final_states = []
                state_seq_type = {}

                for start_state_tuple, final_state_tuple in batch_results.items():
                    start_state = list(start_state_tuple)
                    final_state = list(final_state_tuple)

                    start_state_str = "".join(map(str, start_state))
                    final_states.append("".join(map(str, final_state)))

                    state_scores_dict[start_state_str] = self._CellCycleStateCalculation__calculate_state_scores(
                        final_state
                    )

                    # For JAX processing, we can't track the full state sequence
                    # So mark as JAX-optimized if sequence checking is requested
                    if self._CellCycleStateCalculation__check_sequence:
                        state_seq_type[start_state_str] = "jax_optimized"

                t_end = time.time()
                logging.debug(f"JAX processing completed in {t_end - t_start:.4f} seconds")

                return state_scores_dict, final_states, state_seq_type

            except Exception as e:
                logging.error(f"Error in JAX processing: {e}")
                logging.warning("Falling back to original implementation")

        # Use original implementation
        logging.debug("Using original state processing implementation")
        return super().__iterate_all_start_states(graph_matrix, graph_mod_id)
