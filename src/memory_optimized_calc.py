"""
Memory-optimized JAX acceleration for very large Boolean cell cycle models.

This module extends the JaxCellCycleCalculation class with memory-efficient
processing capabilities for handling extremely large state spaces.
"""

import gc
import logging
import time

from src.jax_state_calc import JAX_AVAILABLE, JaxCellCycleCalculation

# Import memory optimization if available
try:
    from src.utils.jax_memory_optimizer import (
        adaptive_batch_size,
        estimate_memory_requirements,
        memory_efficient_state_processing,
    )

    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False
    logging.warning("Memory optimization not available")


class MemoryOptimizedCalculation(JaxCellCycleCalculation):
    """
    Memory-optimized version of JaxCellCycleCalculation for very large models.

    This class extends JaxCellCycleCalculation with specialized memory handling
    to process extremely large state spaces that might otherwise cause out-of-memory
    errors on GPUs with limited memory.
    """

    def __init__(self, file_inputs, model_specific_inputs, user_inputs):
        """Initialize with memory optimization settings"""
        # Add memory optimization parameters
        self.__memory_limit_gb = file_inputs.get("memory_limit_gb", None)
        self.__use_memory_optimization = file_inputs.get("use_memory_optimization", True)

        # Call parent initialization
        super().__init__(file_inputs, model_specific_inputs, user_inputs)

        # Check if we can use memory optimization
        self.__can_use_memory_optimization = (
            JAX_AVAILABLE and MEMORY_OPTIMIZER_AVAILABLE and self.__use_memory_optimization
        )

        if self.__can_use_memory_optimization:
            # Calculate model complexity
            num_cyclins = len(self._CellCycleStateCalculation__all_cyclins)
            num_states = 2**num_cyclins

            # Estimate memory requirements
            peak_gb, steady_gb = estimate_memory_requirements(num_cyclins, self.batch_size_gpu)

            logging.info(f"Model complexity: {num_cyclins} cyclins, {num_states} states")
            logging.info(f"Memory requirements: Peak={peak_gb:.2f}GB, Steady={steady_gb:.2f}GB")

            # Use adaptive batch size if model is very large
            if num_cyclins > 16:
                self.memory_batch_size = adaptive_batch_size(num_cyclins, self.__memory_limit_gb)
                logging.info(f"Using memory-optimized batch size: {self.memory_batch_size}")
            else:
                self.memory_batch_size = self.batch_size_gpu
        else:
            self.memory_batch_size = self.batch_size_gpu

    def __iterate_all_start_states(self, graph_matrix, graph_mod_id):
        """
        Memory-optimized version of state iteration for very large models.

        This method overrides the parent implementation to use memory-efficient
        processing when dealing with large state spaces.

        Args:
            graph_matrix: The graph matrix representing the network
            graph_mod_id: Identifier for the graph modification

        Returns:
            tuple: (state_scores_dict, final_states, state_seq_type)
        """
        # If memory optimization is not available or not needed, use parent implementation
        if not self.__can_use_memory_optimization:
            return super().__iterate_all_start_states(graph_matrix, graph_mod_id)

        # Check if model is complex enough to warrant memory optimization
        num_cyclins = len(self._CellCycleStateCalculation__all_cyclins)
        if num_cyclins <= 14:  # For smaller models, standard JAX processing is sufficient
            return super().__iterate_all_start_states(graph_matrix, graph_mod_id)

        # From here on, use memory-optimized processing
        logging.info("Using memory-optimized JAX processing")

        # Select states based on configuration
        if self._CellCycleStateCalculation__g1_states_only_flag:
            states_to_process = self._CellCycleStateCalculation__g1_start_states
        else:
            states_to_process = self._CellCycleStateCalculation__start_states

        # Run garbage collection before processing
        gc.collect()

        # Process using memory-efficient JAX acceleration
        t_start = time.time()
        try:
            # Process all states with memory optimization
            batch_results = memory_efficient_state_processing(
                states_to_process,
                graph_matrix,
                batch_size=self.memory_batch_size,
                max_iterations=self._CellCycleStateCalculation__max_iter_count,
                device=self.primary_device,
                memory_limit_gb=self.__memory_limit_gb,
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
                if self._CellCycleStateCalculation__check_sequence:
                    state_seq_type[start_state_str] = "memory_optimized"

            t_end = time.time()
            logging.info(f"Memory-optimized processing completed in {t_end - t_start:.4f} seconds")

            return state_scores_dict, final_states, state_seq_type

        except Exception as e:
            logging.error(f"Error in memory-optimized processing: {e}")
            logging.warning("Falling back to standard JAX implementation")
            return super().__iterate_all_start_states(graph_matrix, graph_mod_id)
