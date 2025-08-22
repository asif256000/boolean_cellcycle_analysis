# Author: Asif Iqbal Rahaman
# Description: This script calculates the state of a system using GPU acceleration.
# Date: 2025-06-06

# Import relevant libraries
from argparse import Namespace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import device_put, jit, vmap

from src.all_inputs import InputTemplate


class CellCycleStateCalculator:
    def __init__(self, file_inputs: dict, model_specific_inputs: InputTemplate, user_inputs: Namespace):
        """
        Initialize the state calculator with file inputs (from the yaml file),
        model-specific inputs (from the `all_inputs` file, which are
        determined and fixed via experiments and literature survery), and
        user inputs (CLI inputs taken as arguments via `argparser`).

        :param dict file_inputs: Inputs from the yaml file.
        :param InputTemplate model_specific_inputs: Model-specific inputs.
        :param Namespace user_inputs: User inputs from CLI arguments.
        """
        self.__all_cyclins = model_specific_inputs.cyclins
        self.__organism = user_inputs.organism

        self.__expected_final_state = model_specific_inputs.expected_final_state
        self.__expected_cyclin_order = model_specific_inputs.expected_cyclin_order
        self.__g1_state_zero_cyclins = model_specific_inputs.g1_state_zero_cyclins
        self.__g1_state_one_cyclins = model_specific_inputs.g1_state_one_cyclins

        self.__optimal_graph_score = model_specific_inputs.optimal_graph_score
        self.__optimal_g1_only_score = model_specific_inputs.g1_only_optimal_graph_score
        self.__self_activation_flag = model_specific_inputs.rule_based_self_activation
        self.__self_inhibition_flag = model_specific_inputs.rule_based_self_deactivation
        self.__cell_cycle_activation_cyclin = model_specific_inputs.cell_cycle_activation_cyclin
        self.__model_graph = model_specific_inputs.modified_graph
        self.__sequence_penalty = self.__calculate_penalty()

        self.__check_sequence = file_inputs.get("check_sequence", True)
        self.__async_update = file_inputs.get("async_update", True)
        self.__random_cyclin_update = file_inputs.get("random_cyclin_update", True)
        self.__complete_cycle = file_inputs.get("complete_cycle", False)
        self.__expensive_cycle_detection = file_inputs.get("expensive_cycle_detection", False)
        self.__max_iterations = file_inputs.get("max_iterations", 50)

        self.__all_states = self.__get_all_possible_starting_states()
        self.__start_states = np.array(self.__all_states, dtype=np.int_)
        # self.__g1_states = self.__get_all_g1_states()

        # JAX Optimization Parameters
        self.__use_gpu = file_inputs.get("use_gpu", True)
        self.__force_cpu_only = file_inputs.get("force_cpu_only", False)
        self.__configure_jax()

    def __repr__(self) -> str:
        """
        String representation of the CellCycleStateCalculator object.
        """
        return (
            f"CellCycleStateCalculator(Organism:{self.__organism}, "
            f"Cyclins:{self.__all_cyclins}, "
            f"Expected Final State:{dict(zip(self.__all_cyclins, self.__expected_final_state))}, "
        )

    def __configure_jax(self):
        """
        Configure JAX to use GPU if available and not forced to use CPU.
        """
        all_devices = jax.devices()
        self.__gpu_devices = list()
        self.__cpu_devices = list()

        for device in all_devices:
            if device.platform == "gpu" and self.__use_gpu and not self.__force_cpu_only:
                self.__gpu_devices.append(device)
            elif device.platform == "cpu" or self.__force_cpu_only:
                self.__cpu_devices.append(device)

        # Set primary device based on user preference
        if self.__force_cpu_only:
            jax.config.update("jax_platform_name", "cpu")
            # Ensure that the CPU device is actually selected
            if not self.__cpu_devices:
                raise RuntimeError("No CPU devices found, but force_cpu_only is True.")
            jax.config.update("jax_default_device", self.__cpu_devices[0])
        elif self.__use_gpu and self.__gpu_devices:
            jax.config.update("jax_platform_name", "gpu")
            jax.config.update("jax_default_device", self.__gpu_devices[0])
        else:
            # Fallback to CPU if GPU is not used or not available
            jax.config.update("jax_platform_name", "cpu")
            jax.config.update("jax_default_device", self.__cpu_devices[0])

        self.__gpu_batch_size = min(len(self.__gpu_devices) * 512, len(self.__all_cyclins) ** 2)
        print(f"Using devices: {jax.devices()}")

    def __calculate_penalty(self) -> int:
        """
        Calculate the penalty based on the number of nodes in the graph
        for which the state is fixed to 0 or 1 in the G1 state.

        :return int: The penalty value calculated as 2 raised to the power of the
        """
        return 2 ** (len(self.__g1_state_zero_cyclins) + len(self.__g1_state_one_cyclins))

    def __get_all_possible_starting_states(self) -> list[list]:
        """
        Generate all possible starting states for the cyclins based on the
        G1 state zero and one cyclins.

        :return list[list]: A list of lists containing all possible starting states.
        """
        cyclin_count = len(self.__all_cyclins)
        complete_state_list = [f"{i:>0{cyclin_count}b}" for i in range(2**cyclin_count)]

        return [list(map(int, list(state))) for state in complete_state_list]

    # def __get_all_g1_states(self) -> list[list]:
    #     """
    #     Generate all possible G1 states based on the G1 state zero and one cyclins.

    #     Returns:
    #         list[list]: A list of lists containing all possible G1 states.
    #     """
    #     return self.filter_starting_states(self.__g1_state_zero_cyclins, self.__g1_state_one_cyclins)

    def __get_cyclin_index(self, cyclin: str) -> int:
        """
        Get the index of a cyclin in the list of all cyclins.
        This method raises a ValueError if the cyclin is not found in the list.

        :param cyclin (str): The name of the cyclin.
        :raises ValueError: If the cyclin is not found in the list of all cyclins.
        :returns: int: The index of the cyclin in the list.
        """
        try:
            return self.__all_cyclins.index(cyclin)
        except ValueError:
            raise ValueError(f"Cyclin '{cyclin}' not found in the list of all cyclins.")

    def _get_all_possible_starting_states(self) -> npt.NDArray[np.int_]:
        """
        Generate all possible starting states for the cyclins.

        :return npt.NDArray[np.int_]: A 2D numpy array where each row represents a starting state.
        """
        num_of_cyclins = len(self.__all_cyclins)
        total_states = 2**num_of_cyclins
        states_array = np.arange(total_states, dtype=np.uint16).reshape(-1, 1)
        # Use bitwise operations to create the binary states
        return ((states_array >> np.arange(num_of_cyclins - 1, -1, -1)) & 1).astype(np.int_)

    def set_starting_states(self, starting_states: npt.NDArray[np.int_]) -> None:
        """
        Set the starting states for the cyclins.

        :param npt.NDArray[np.int_] starting_states: A 2D array where each row represents a starting state.
        """
        if not isinstance(starting_states, np.ndarray) or starting_states.ndim != 2:
            raise TypeError("Starting states must be a 2D numpy array.")
        self.__start_states = starting_states

    def filter_starting_states(self, zero_cyclins: list[str], one_cyclins: list[str]) -> None:
        """
        Filter the starting states, keeping nodes fixed to zero or one.

        :param list[str] zero_cyclins: List of cyclins to be fixed to 0.
        :param list[str] one_cyclins: List of cyclins to be fixed to 1.
        """
        zero_ixs = [self.__get_cyclin_index(cyclin) for cyclin in zero_cyclins]
        one_ixs = [self.__get_cyclin_index(cyclin) for cyclin in one_cyclins]

        filtered_states = []
        for state in self.__start_states:
            if all(state[ix] == 0 for ix in zero_ixs) and all(state[ix] == 1 for ix in one_ixs):
                filtered_states.append(state)
        self.__start_states = np.array(filtered_states, dtype=np.int_)

    @partial(jit, static_argnums=(0,))
    def _async_calculate_next_state(self, graph_matrix: jnp.ndarray, current_state: jnp.ndarray, cyclin_ix: int) -> jnp.ndarray:
        """
        Calculate the next state based on the current state of the cyclins.

        :param jnp.ndarray graph_matrix: The interaction graph matrix.
        :param jnp.ndarray current_state: A 2D array where each row represents the current state of the cyclins.
        :param int cyclin_ix: The index of the cyclin to update.
        :returns: jnp.ndarray: The next state of the cyclins.
        """
        next_state = current_state.copy()

        # Get the row for the specific cyclin (incoming edges to this cyclin)
        cyclin_row = graph_matrix[cyclin_ix]

        # Set self-loop to zero to exclude it from calculation
        cyclin_row_no_self = cyclin_row.at[cyclin_ix].set(0)

        # Calculate state value using dot product: sum(edge_weight * source_state)
        state_values = jnp.dot(current_state, cyclin_row_no_self)

        # Apply Boolean logic for the specific cyclin
        new_cyclin_states = jnp.where(state_values > 0, 1, jnp.where(state_values < 0, 0, current_state[:, cyclin_ix]))

        # Apply self-loop rules if no change occurred
        no_change_mask = new_cyclin_states == current_state[:, cyclin_ix]
        new_cyclin_states = self._apply_async_self_loops(
            graph_matrix, current_state, new_cyclin_states, cyclin_ix, no_change_mask
        )

        # Update only the specific cyclin in the next state
        next_state = next_state.at[:, cyclin_ix].set(new_cyclin_states)

        return next_state

    @partial(jit, static_argnums=(0, 4))
    def _apply_async_self_loops(
        self,
        graph_matrix: jnp.ndarray,
        current_state: jnp.ndarray,
        new_cyclin_states: jnp.ndarray,
        cyclin_ix: int,
        no_change_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply self-loop rules for asynchronous updates on a specific cyclin.

        :param jnp.ndarray graph_matrix: The interaction graph matrix
        :param jnp.ndarray current_state: Current state of all cyclins
        :param jnp.ndarray new_cyclin_states: Calculated new states for the specific cyclin
        :param int cyclin_ix: Index of the cyclin being updated
        :param jnp.ndarray no_change_mask: Boolean mask for states with no change
        :return jnp.ndarray: Updated cyclin states with self-loop rules applied
        """
        # Create a mask to exclude the self-loop and select incoming edges
        mask = jnp.arange(graph_matrix.shape[1]) != cyclin_ix
        incoming_edges = graph_matrix[cyclin_ix, mask]

        red_count = jnp.sum(incoming_edges == -1)
        green_count = jnp.sum(incoming_edges == 1)

        # Apply self-inhibition rule
        if self.__self_inhibition_flag:
            inhibition_condition = (red_count == 0) | (green_count > red_count)
            apply_inhibition = no_change_mask & inhibition_condition
            new_cyclin_states = jnp.where(apply_inhibition, 0, new_cyclin_states)

        # Apply self-activation rule
        if self.__self_activation_flag:
            activation_condition = (green_count == 0) | (red_count > green_count)
            apply_activation = no_change_mask & activation_condition
            new_cyclin_states = jnp.where(apply_activation, 1, new_cyclin_states)

        return new_cyclin_states

    @partial(jit, static_argnums=(0,))
    def _sync_calculate_next_state(self, graph_matrix: jnp.ndarray, current_state: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the next state for all cyclins synchronously using vectorized operations.

        :param jnp.ndarray graph_matrix: The interaction graph matrix (n_cyclins x n_cyclins)
        :param jnp.ndarray current_state: Current state of all cyclins (batch_size x n_cyclins)
        :return jnp.ndarray: Next state for all cyclins (batch_size x n_cyclins)
        """
        # Remove self-loops from the graph matrix for calculation
        graph_no_self = graph_matrix - jnp.diag(jnp.diag(graph_matrix))

        # Calculate state values for all cyclins simultaneously
        state_values = jnp.matmul(current_state, graph_no_self.T)

        # Apply Boolean logic for all cyclins
        next_state = jnp.where(state_values > 0, 1, jnp.where(state_values < 0, 0, current_state))

        # Apply self-loop rules for positions with no change
        no_change_mask = next_state == current_state
        next_state = self._apply_sync_self_loops(graph_matrix, current_state, next_state, no_change_mask)

        return next_state

        

    

    def __calculate_state_scores(self, final_state: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates a score by comparing the 'final_state' with the expected
        final state (__expected_final_state) for each cyclin. The score is the sum of absolute
        differences between the two.

        :param jnp.ndarray final_state: The final state of the cyclins.
        :return jnp.ndarray: The calculated score.
        """
        # Convert expected_final_state to a JAX array for computation
        expected_final_state_jnp = jnp.array(self.__expected_final_state, dtype=jnp.int_)

        # Create a mask for integer values in expected_final_state
        # JAX does not have isinstance, so we assume if it's not -1, it's an int
        # This needs to be handled carefully if expected_final_state can contain other non-int types
        # For now, assuming -1 is the only non-integer sentinel value
        is_int_mask = (expected_final_state_jnp != -1)

        # Calculate absolute differences only for integer values
        score_diffs = jnp.abs(final_state - expected_final_state_jnp)
        score = jnp.sum(jnp.where(is_int_mask, score_diffs, 0))
        return score

    def __generate_final_state_counts(self, final_states: jnp.ndarray) -> dict:
        """
        Generates a dictionary with counts of each unique final state.

        :param jnp.ndarray final_states: A 2D array of final states.

        :return dict: A dictionary where keys are string representations of states and values are their counts.

        """
        # Convert JAX array to NumPy array for easier unique counting and dictionary creation
        final_states_np = np.asarray(final_states)
        unique_states, counts = np.unique(final_states_np, axis=0, return_counts=True)

        # Convert unique states (numpy arrays) to string representations for dictionary keys
        final_state_counts_dict = {
            "".join(map(str, state.tolist())): int(count)
            for state, count in zip(unique_states, counts)
        }
        return final_state_counts_dict

    @partial(jit, static_argnums=(0,))
    def __detect_any_cycles(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        Should return True iff there is an ongoing cycle in the second half of the list.
        Cycle here can be defined as a repeated sequence that appears in the list in the same order
        multiple times consecutively.

        :param jnp.ndarray all_states: The 3D array of states in which cycle has to be detected.
        :return jnp.ndarray: Returns a boolean array, True iff a cycle is detected in the second half of the list, False otherwise.
        """
        # Remove continuous duplicates first
        filtered_states = self.remove_continuous_duplicates(all_states)

        # JAX does not support dynamic list slicing or iteration over arbitrary lengths easily
        # This implementation will be a simplified version or require fixed-size assumptions
        # For now, let's assume we are looking for a cycle of a fixed length, or a simplified check

        # A more robust cycle detection in JAX would involve a fixed number of look-back steps
        # or a more complex scan operation. For simplicity, let's check for a cycle of length 2
        # at the end of the sequence, which is a common pattern.

        # This is a placeholder for a more sophisticated JAX-compatible cycle detection.
        # It checks if the last state is equal to the state before the last, which is a very basic cycle.
        # For more complex cycles, a different approach is needed.
        if filtered_states.shape[0] < 2:
            return jnp.full(all_states.shape[1], False, dtype=jnp.bool_)

        # Check if the last state is equal to the second to last state for each batch
        return jnp.all(filtered_states[-1] == filtered_states[-2], axis=1)

    @partial(jit, static_argnums=(0,))
    def __detect_end_cycles(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        Should return True iff there is an ongoing cycle at the end of the list.
        Cycle here can be defined as a repeated sequence that appears in the list
        in the same order multiple times consecutively.

        :param jnp.ndarray all_states: The 3D array of states in which cycle has to be detected.
        :return jnp.ndarray: Returns a boolean array, True iff a cycle is detected at the end of the list, False otherwise.
        """
        filtered_states = self.remove_continuous_duplicates(all_states)
        states_len = filtered_states.shape[0]
        batch_size = filtered_states.shape[1]

        # Initialize a boolean array to track cycle detection for each batch
        cycle_detected = jnp.full(batch_size, False, dtype=jnp.bool_)

        # Iterate through possible cycle lengths (from 2 up to half the length of filtered_states)
        # JAX requires fixed-size loops for `fori_loop` or `scan`.
        # We'll use a fixed maximum cycle length for this example, or iterate up to a reasonable limit.
        # For a general solution, this might need a `scan` over possible lengths.
        max_cycle_len = states_len // 2 + 1

        # Using a for loop for demonstration, but for performance in JAX, consider jax.lax.scan
        # or a fixed number of checks.
        for i in range(2, max_cycle_len):
            if states_len >= 2 * i:
                # Check if the last `i` states are equal to the `i` states before them
                # for each batch independently
                is_cycle_for_len_i = jnp.all(
                    filtered_states[-i:] == filtered_states[-2 * i : -i],
                    axis=(0, 2)  # Compare across iterations and cyclins
                )
                cycle_detected = cycle_detected | is_cycle_for_len_i

        return cycle_detected

    @partial(jit, static_argnums=(0,))
    def __check_activation_index(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        This method examines the list of states generated during the simulation to find
        the index at which a specified cyclin is activated. The activation is determined
        by the presence of a value '1' in the cyclin's position in the state vector.

        :param jnp.ndarray all_states: A 3D array of states (iterations, batch_size, num_cyclins).
        :return jnp.ndarray: An array of integers, where each element is the activation index for a batch.
                             Returns -1 if the cyclin is already active at the start or never activates.
        """
        target_cyclin_ix = self.__get_cyclin_index(self.__cell_cycle_activation_cyclin)
        filtered_states = self.remove_continuous_duplicates(all_states)

        # Check if the target cyclin is already 1 at the initial state for each batch
        initial_activation = (filtered_states[0, :, target_cyclin_ix] == 1)

        # Find the first index where the target cyclin becomes 1
        # We need to iterate through the states and find the first occurrence for each batch
        # This can be done efficiently using argmax with a mask.
        # Create a mask where the target cyclin is 1
        activation_mask = (filtered_states[:, :, target_cyclin_ix] == 1)

        # Find the first index where activation_mask is True along the iteration axis
        # jnp.argmax returns the first index of the maximum value. If all are False, it returns 0.
        # We need to handle the case where it never activates.
        # To do this, we can add a large value to non-activating states so argmax doesn't pick them.
        # Or, more simply, use a cumulative sum to find the first True.

        # Create an array of indices for iterations
        indices = jnp.arange(filtered_states.shape[0])

        # For each batch, find the first index where activation_mask is True
        # If no True, the sum will be 0, and we can map it to -1
        # This approach assumes that once activated, it stays activated.
        activated_indices = jnp.where(
            activation_mask,
            indices[:, None],  # Broadcast indices to match batch_size
            filtered_states.shape[0] + 1  # A value larger than any valid index
        ).min(axis=0) # Find the minimum index (first activation)

        # If activated_indices is still filtered_states.shape[0] + 1, it means no activation occurred.
        # Map this to -1.
        activated_indices = jnp.where(
            activated_indices == filtered_states.shape[0] + 1,
            -1,
            activated_indices
        )

        # If initially activated, set to -1
        activated_indices = jnp.where(initial_activation, -1, activated_indices)

        return activated_indices

    @partial(jit, static_argnums=(0,))
    def verify_sequence(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        Checks if the state path that was traversed follows expected order.
        Returns True if it follows correct sequence, otherwise returns False.

        :param jnp.ndarray all_states: A 3D array of states (iterations, batch_size, num_cyclins).
        :return jnp.ndarray: A boolean array indicating for each batch if the sequence was followed.
        """
        filtered_states = self.remove_continuous_duplicates(all_states)
        batch_size = filtered_states.shape[1]
        num_cyclins = filtered_states.shape[2]

        # Initialize sequence_followed to True for all batches
        sequence_followed = jnp.full(batch_size, True, dtype=jnp.bool_)

        # Convert expected_cyclin_order to a JAX array of dictionaries (or a structured array)
        # This is tricky with JAX, as it prefers numerical arrays.
        # We'll need to convert the expected_cyclin_order (list of dicts) into a more JAX-friendly format.
        # For now, let's assume expected_cyclin_order is a list of arrays or a 2D array.
        # If it's a list of dictionaries, we'll need to convert it to a fixed-size JAX array
        # where each row represents a state in the expected sequence.

        # Example: Convert expected_cyclin_order to a JAX array of expected states
        # This assumes a fixed order and structure for the expected states.
        # For a more general solution, this part needs careful consideration.
        expected_order_jnp = jnp.array(
            [[state.get(cyclin, -1) for cyclin in self.__all_cyclins] for state in self.__expected_cyclin_order],
            dtype=jnp.int_
        )

        # Iterate through the filtered states and check against the expected order
        # This will be a nested loop or a scan operation.
        # For simplicity, let's use a for loop for now, but be aware of JAX's limitations.

        # We need to track the current position in the expected_order for each batch.
        # Initialize current_expected_idx for each batch
        current_expected_idx = jnp.zeros(batch_size, dtype=jnp.int_)

        def scan_body(carry, current_state_batch):
            current_expected_idx, sequence_followed = carry

            # For each batch, check if the current state matches the expected state
            # at current_expected_idx
            def check_match(idx, state, expected_idx, seq_followed_single):
                # If sequence already failed for this batch, keep it failed
                if not seq_followed_single:
                    return expected_idx, False

                # Get the expected state for the current expected_idx
                # Handle out-of-bounds for expected_order_jnp if all expected states are matched
                expected_state = jnp.where(
                    expected_idx < expected_order_jnp.shape[0],
                    expected_order_jnp[expected_idx],
                    jnp.full(num_cyclins, -2, dtype=jnp.int_) # Sentinel value for already matched
                )

                # Check if the current state matches the expected state (considering -1 as wildcard)
                # A state matches if for all cyclins, either the expected value is -1 (wildcard)
                # or the current state value matches the expected value.
                match = jnp.all(
                    jnp.where(expected_state == -1, True, state == expected_state)
                )

                # If matched, advance expected_idx
                new_expected_idx = jnp.where(match, expected_idx + 1, expected_idx)
                new_seq_followed_single = jnp.where(match, True, False) # If no match, sequence fails

                return new_expected_idx, new_seq_followed_single

            # Vectorize the check_match function over the batch dimension
            new_expected_idx_batch, new_sequence_followed_batch = vmap(check_match)(
                jnp.arange(batch_size), current_state_batch, current_expected_idx, sequence_followed
            )

            return (new_expected_idx_batch, new_sequence_followed_batch), None

        # Perform the scan over the filtered states
        (final_expected_idx, final_sequence_followed), _ = jax.lax.scan(
            scan_body,
            (current_expected_idx, sequence_followed),
            filtered_states
        )

        # The sequence is followed if the final_expected_idx has reached the end of expected_order_jnp
        # for each batch, and sequence_followed is still True.
        # If final_expected_idx is less than the length of expected_order_jnp, it means not all
        # expected states were found.
        sequence_followed_final = final_sequence_followed & (final_expected_idx >= expected_order_jnp.shape[0])

        return sequence_followed_final

    @partial(jit, static_argnums=(0,))
    def __iterate_all_start_states(
        self,
        graph_matrix: jnp.ndarray,
        start_states: jnp.ndarray,
        random_key: jax.random.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
        """
        This method iterates through all possible start states of the cell cycle model and
        simulates the cell cycle dynamics. It calculates state scores and determines the
        final states of each simulation.

        :param jnp.ndarray graph_matrix: The interaction graph matrix.
        :param jnp.ndarray start_states: A 2D array of initial states.
        :param jax.random.PRNGKey random_key: JAX random key for reproducibility.
        :returns: tuple[jnp.ndarray, jnp.ndarray, dict]: A tuple containing:
                  - state_scores: A 1D array of scores for each starting state.
                  - final_states: A 2D array of final states for each starting state.
                  - state_seq_type: A dictionary mapping string representation of start states to their sequence type.
        """
        batch_size = start_states.shape[0]
        num_cyclins = len(self.__all_cyclins)

        # Generate state tables for all starting states in parallel
        all_cyclin_states, _ = self.__generate_state_table(
            graph_matrix, start_states, random_key
        )

        # Initialize final_states and state_scores
        final_states = jnp.zeros((batch_size, num_cyclins), dtype=jnp.int_)
        state_scores = jnp.zeros(batch_size, dtype=jnp.int_)

        # Cycle detection logic
        # Determine which states have cycles based on expensive_cycle_detection flag
        if self.__expensive_cycle_detection:
            cycle_detected_mask = self.__detect_end_cycles(all_cyclin_states)
        else:
            cycle_detected_mask = self.__lazy_detect_cycles(all_cyclin_states)

        # Handle states with cycles
        # For states with cycles, final_state is represented by a special value (e.g., -1)
        # and score is calculated from the last two states.
        cycle_final_state_val = jnp.full(num_cyclins, -1, dtype=jnp.int_) # Using -1 to denote a cycle
        cycle_score = vmap(self.__calculate_state_scores)(all_cyclin_states[-1]) + vmap(self.__calculate_state_scores)(all_cyclin_states[-2])

        final_states = jnp.where(
            cycle_detected_mask[:, None], # Add new axis for broadcasting
            cycle_final_state_val,
            all_cyclin_states[-1]
        )
        state_scores = jnp.where(
            cycle_detected_mask,
            cycle_score,
            vmap(self.__calculate_state_scores)(all_cyclin_states[-1])
        )

        # Initialize sequence type arrays
        # 0: did_not_start, 1: correct, 2: incorrect
        state_seq_type_int = jnp.full(batch_size, -1, dtype=jnp.int_)

        # Check activation index for all batches
        activation_indices = self.__check_activation_index(all_cyclin_states)
        cell_div_start_flag = (activation_indices != -1)

        # Determine if start states are G1 states (assuming self.__g1_start_states is a JAX array)
        # This requires converting self.__g1_start_states to a JAX array if it's not already.
        # For now, let's assume self.__g1_start_states is available as a JAX array or can be converted.
        # If self.__g1_start_states is a list of lists, convert it to a JAX array:
        g1_start_states_jnp = jnp.array(self.__get_all_g1_states(), dtype=jnp.int_)

        # Check if each start_state is in g1_start_states_jnp
        is_g1_state = jnp.any(jnp.all(start_states[:, None, :] == g1_start_states_jnp[None, :, :], axis=2), axis=1)

        # Apply sequence verification and penalty
        if self.__check_sequence:
            # Verify sequence for all batches
            sequence_followed = self.verify_sequence(all_cyclin_states)

            # Update state_seq_type_int based on conditions
            # If not cell_div_start_flag
            did_not_start_mask = ~cell_div_start_flag
            state_seq_type_int = jnp.where(did_not_start_mask, 0, state_seq_type_int)
            state_scores = jnp.where(did_not_start_mask & is_g1_state, state_scores + self.__sequence_penalty, state_scores)

            # If cell_div_start_flag and sequence_followed
            correct_seq_mask = cell_div_start_flag & sequence_followed
            state_seq_type_int = jnp.where(correct_seq_mask, 1, state_seq_type_int)

            # If cell_div_start_flag and not sequence_followed
            incorrect_seq_mask = cell_div_start_flag & ~sequence_followed
            state_seq_type_int = jnp.where(incorrect_seq_mask, 2, state_seq_type_int)
            state_scores = jnp.where(incorrect_seq_mask & is_g1_state, state_scores + self.__sequence_penalty, state_scores)

        # Convert state_seq_type_int to a dictionary of string representations
        state_seq_type_map = {
            0: "did_not_start",
            1: "correct",
            2: "incorrect",
            -1: "unknown" # For states not covered by the above conditions
        }
        state_seq_type_dict = {
            "".join(map(str, start_states[i].tolist())):
            state_seq_type_map[state_seq_type_int[i].item()]
            for i in range(batch_size)
        }

        return state_scores, final_states, state_seq_type_dict

    def __get_all_g1_states(self) -> list[list]:
        """
        Generate all possible G1 states based on the G1 state zero and one cyclins.

        Returns:
            list[list]: A list of lists containing all possible G1 states.
        """
        # Temporarily set the start states to all possible states to filter G1 states
        original_start_states = self.__start_states
        self.__start_states = self._get_all_possible_starting_states()
        self.filter_starting_states(self.__g1_state_zero_cyclins, self.__g1_state_one_cyclins)
        g1_states = self.__start_states.tolist()
        self.__start_states = original_start_states # Restore original start states
        return g1_states

    def generate_graph_score_and_final_states(
        self, graph_matrix: jnp.ndarray, graph_mod_id: str
    ) -> tuple[jnp.ndarray, dict, dict]:
        """
        This method calculates a graph score and generates final state information for a
        given graph modification, which includes the network's adjacency matrix and a
        unique identifier. The graph score is determined based on state scores, and the
        final state information includes counts and sequence types.

        :param jnp.ndarray graph_matrix: The interaction graph matrix.
        :param str graph_mod_id: A unique identifier for the graph modification.
        :returns: tuple[jnp.ndarray, dict, dict]: A tuple containing:
                  - graph_score: The total graph score.
                  - final_state_count: A dictionary with counts of each unique final state.
                  - state_seq_types: A dictionary mapping string representation of start states to their sequence type.
        """
        # Generate a new random key for each simulation run
        key = jax.random.PRNGKey(0) # TODO: Make this dynamic or pass from outside

        state_scores, final_states, state_seq_types = self.__iterate_all_start_states(
            graph_matrix, self.__start_states, key
        )
        final_state_count = self.__generate_final_state_counts(final_states)
        graph_score = jnp.sum(state_scores)

        return graph_score, final_state_count, state_seq_types

    def print_final_state_count_table(self, final_state_count: dict, log_level: str = "debug"):
        """
        Prints a formatted table of final state counts.

        :param dict final_state_count: A dictionary where keys are string representations of states and values are their counts.
        :param str log_level: The logging level to use (e.g., "debug", "info").
        """
        table_as_str = "Count of final state for each start states:\n"
        for state_str, count in final_state_count.items():
            # Convert state_str back to a list of ints for state_as_str
            state_list = [int(x) for x in list(state_str)]
            table_as_str += f"Count: {count:>05}, "
            table_as_str += self.state_as_str(state_list)
            table_as_str += "\n"

        # Assuming a logger is available (e.g., from src.log_module)
        # if log_level.lower() == "info":
        #     logger.info(table_as_str)
        # else:
        #     logger.debug(table_as_str)
        print(table_as_str) # For now, just print

    def state_as_str(self, state: list) -> str:
        """
        Converts a state (list of ints) into a human-readable string representation.

        :param list state: A list of integers representing the state of cyclins.
        :return str: A string representation of the state.
        """
        state_dict = dict(zip(self.__all_cyclins, state))
        return ", ".join([f"{k}: {v}" for k, v in state_dict.items()])

    def print_state_table(self, cyclin_states: jnp.ndarray, random_update_sequence: jnp.ndarray, log_level: str = "debug"):
        """
        Prints a formatted table of state changes over time for a single simulation.

        :param jnp.ndarray cyclin_states: A 3D array of states (iterations, batch_size=1, num_cyclins).
        :param jnp.ndarray random_update_sequence: A 2D array of cyclin indices updated at each step (iterations, batch_size=1).
        :param str log_level: The logging level to use (e.g., "debug", "info").
        """
        # Assuming cyclin_states and random_update_sequence are for a single batch (batch_size=1)
        # Squeeze to remove the batch dimension for easier processing
        cyclin_states_np = np.asarray(cyclin_states).squeeze(axis=1)
        random_update_sequence_np = np.asarray(random_update_sequence).squeeze(axis=1)

        curr_tracked_state = cyclin_states_np[0].tolist()
        table_as_str = f"State sequence for start state: {self.state_as_str(curr_tracked_state)}\n"
        table_as_str += "Time: 0000, " + self.state_as_str(curr_tracked_state)
        if random_update_sequence_np.size > 0:
            table_as_str += ", Update: Start\n"
        else:
            table_as_str += "\n"

        for ix, ix_state_np in enumerate(cyclin_states_np[1:]):
            ix_state = ix_state_np.tolist()
            # if self.__view_state_change_only and ix_state == curr_tracked_state:
            #     continue # This logic needs to be handled carefully with JAX arrays

            table_as_str += f"Time: {ix + 1:>04}, "
            table_as_str += self.state_as_str(ix_state)
            if random_update_sequence_np.size > 0 and ix < random_update_sequence_np.size:
                updated_cyclin_name = self.__all_cyclins[random_update_sequence_np[ix]]
                table_as_str += f", Update: {updated_cyclin_name}\n"
            else:
                table_as_str += "\n"
            curr_tracked_state = ix_state

        # Assuming a logger is available (e.g., from src.log_module)
        # if log_level.lower() == "info":
        #     logger.info(table_as_str)
        # else:
        #     logger.debug(table_as_str)
        print(table_as_str) # For now, just print

    @partial(jit, static_argnums=(0, 1, 2))
    def __generate_state_table(
        self,
        graph_matrix: jnp.ndarray,
        start_states: jnp.ndarray,
        random_key: jax.random.PRNGKey,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generates a state table by simulating the dynamics of the network
        model described by the given graph matrix, starting from the provided
        initial states.

        :param jnp.ndarray graph_matrix: The interaction graph matrix.
        :param jnp.ndarray start_states: A 2D array of initial states.
        :param jax.random.PRNGKey random_key: JAX random key for reproducibility.
        :returns: tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the sequence of states
                  and the update order (if asynchronous and random).
        """
        num_cyclins = len(self.__all_cyclins)
        batch_size = start_states.shape[0]

        # Initialize state sequence with starting states
        all_cyclin_states = jnp.zeros(
            (self.__max_iterations + 1, batch_size, num_cyclins), dtype=jnp.int_
        )
        all_cyclin_states = all_cyclin_states.at[0].set(start_states)

        # Initialize update order tracker
        update_order = jnp.zeros(
            (self.__max_iterations, batch_size), dtype=jnp.int_
        )

        current_state = start_states

        def async_loop_body(i, loop_state):
            current_state, all_cyclin_states, update_order, key = loop_state
            key, subkey = jax.random.split(key)

            if self.__random_cyclin_update:
                if self.__complete_cycle:
                    # Shuffle cyclins for each full cycle
                    shuffled_indices = jax.random.permutation(subkey, jnp.arange(num_cyclins))
                    cyclin_ix = shuffled_indices[i % num_cyclins]
                else:
                    cyclin_ix = jax.random.randint(subkey, (), 0, num_cyclins)
            else:
                cyclin_ix = i % num_cyclins

            next_state = self._async_calculate_next_state(
                graph_matrix, current_state, cyclin_ix
            )
            all_cyclin_states = all_cyclin_states.at[i + 1].set(next_state)
            update_order = update_order.at[i].set(cyclin_ix)
            return next_state, all_cyclin_states, update_order, key

        def sync_loop_body(i, loop_state):
            current_state, all_cyclin_states, update_order, key = loop_state
            next_state = self._sync_calculate_next_state(graph_matrix, current_state)
            all_cyclin_states = all_cyclin_states.at[i + 1].set(next_state)
            return next_state, all_cyclin_states, update_order, key

        if self.__async_update:
            final_state, all_cyclin_states, update_order, _ = jax.lax.fori_loop(
                0, self.__max_iterations, async_loop_body,
                (current_state, all_cyclin_states, update_order, random_key)
            )
        else:
            final_state, all_cyclin_states, update_order, _ = jax.lax.fori_loop(
                0, self.__max_iterations, sync_loop_body,
                (current_state, all_cyclin_states, update_order, random_key)
            )

        return all_cyclin_states, update_order

    @partial(jit, static_argnums=(0,))
    def remove_continuous_duplicates(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        Removes continuous duplicate states from a sequence of states.

        :param jnp.ndarray all_states: A 3D array of states (iterations, batch_size, num_cyclins).
        :return jnp.ndarray: A 3D array with continuous duplicates removed.
        """
        # Compare each state with the previous state along the iteration axis
        # We need to handle the first state separately as it has no previous state
        is_duplicate = jnp.concatenate([
            jnp.array([False], dtype=jnp.bool_).reshape(1, 1, 1),
            jnp.all(all_states[1:] == all_states[:-1], axis=(2), keepdims=True)
        ], axis=0)

        # Invert the mask to keep non-duplicate states
        non_duplicate_mask = ~is_duplicate.squeeze(axis=(1, 2))

        # Use boolean indexing to filter out duplicate states
        # This will return a 2D array, so we need to reshape it back to 3D if necessary
        filtered_states = all_states[non_duplicate_mask]

        return filtered_states

    @partial(jit, static_argnums=(0,))
    def __lazy_detect_cycles(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        This method checks if a cyclic pattern has been detected in the list of states
        generated during the simulation. The detection depends on whether the simulation
        is asynchronous or synchronous.

        :param jnp.ndarray all_states: A 3D array of states (iterations, batch_size, num_cyclins).
        :return jnp.ndarray: A boolean array indicating for each batch if a cycle is detected.
        """
        if not self.__async_update:
            # For synchronous updates, a cycle is detected if the last two states are different
            # This is a 'lazy' detection, meaning it just checks if the state is still changing
            return jnp.any(all_states[-1] != all_states[-2], axis=1)
        else:
            # For asynchronous updates, a cycle is detected if the last state is different
            # from the state 'num_cyclins' steps back (representing a full cycle of updates)
            return jnp.any(all_states[-1] != all_states[-1 * len(self.__all_cyclins)], axis=1)

    @partial(jit, static_argnums=(0,))
    def __detect_any_cycles(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        Should return True iff there is an ongoing cycle in the second half of the list.
        Cycle here can be defined as a repeated sequence that appears in the list in the same order
        multiple times consecutively.

        :param jnp.ndarray all_states: The 3D array of states in which cycle has to be detected.
        :return jnp.ndarray: Returns a boolean array, True iff a cycle is detected in the second half of the list, False otherwise.
        """
        # Remove continuous duplicates first
        filtered_states = self.remove_continuous_duplicates(all_states)

        # JAX does not support dynamic list slicing or iteration over arbitrary lengths easily
        # This implementation will be a simplified version or require fixed-size assumptions
        # For now, let's assume we are looking for a cycle of a fixed length, or a simplified check

        # A more robust cycle detection in JAX would involve a fixed number of look-back steps
        # or a more complex scan operation. For simplicity, let's check for a cycle of length 2
        # at the end of the sequence, which is a common pattern.

        # This is a placeholder for a more sophisticated JAX-compatible cycle detection.
        # It checks if the last state is equal to the second to last state for each batch.
        if filtered_states.shape[0] < 2:
            return jnp.full(all_states.shape[1], False, dtype=jnp.bool_)

        return jnp.all(filtered_states[-1] == filtered_states[-2], axis=1)

    @partial(jit, static_argnums=(0,))
    def __detect_end_cycles(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        Should return True iff there is an ongoing cycle at the end of the list.
        Cycle here can be defined as a repeated sequence that appears in the list
        in the same order multiple times consecutively.

        :param jnp.ndarray all_states: The 3D array of states in which cycle has to be detected.
        :return jnp.ndarray: Returns a boolean array, True iff a cycle is detected at the end of the list, False otherwise.
        """
        filtered_states = self.remove_continuous_duplicates(all_states)
        states_len = filtered_states.shape[0]
        batch_size = filtered_states.shape[1]

        # Initialize a boolean array to track cycle detection for each batch
        cycle_detected = jnp.full(batch_size, False, dtype=jnp.bool_)

        # Iterate through possible cycle lengths (from 2 up to half the length of filtered_states)
        # JAX requires fixed-size loops for `fori_loop` or `scan`.
        # We'll use a fixed maximum cycle length for this example, or iterate up to a reasonable limit.
        # For a general solution, this might need a `scan` over possible lengths.
        max_cycle_len = states_len // 2 + 1

        # Using a for loop for demonstration, but for performance in JAX, consider jax.lax.scan
        # or a fixed number of checks.
        for i in range(2, max_cycle_len):
            if states_len >= 2 * i:
                # Check if the last `i` states are equal to the `i` states before them
                # for each batch independently
                is_cycle_for_len_i = jnp.all(
                    filtered_states[-i:] == filtered_states[-2 * i : -i],
                    axis=(0, 2)  # Compare across iterations and cyclins
                )
                cycle_detected = cycle_detected | is_cycle_for_len_i

        return cycle_detected