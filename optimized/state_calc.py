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
        self.__start_states = self.__all_states
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
        if self.__use_gpu and not self.__force_cpu_only and self.__gpu_devices:
            jax.config.update("jax_platform_name", "gpu")
            jax.config.update("jax_default_device", self.__gpu_devices[0])
        else:
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

    def get_all_possible_starting_states(self) -> npt.NDArray:
        """
        Get all possible starting states for the cyclins.
        This method returns a 2D numpy array where each row represents a starting state.

        :return npt.NDArray: A 2D numpy array where each row represents a starting state.
        """
        num_of_cyclins = len(self.__all_cyclins)
        total_states = 2**num_of_cyclins

        # Create a 2D array directly
        states_array = np.zeros((total_states, num_of_cyclins), dtype=np.int8)

        # Fill the array with binary representations
        for i in range(total_states):
            # Convert integer to binary and get digits
            binary = format(i, f"0{num_of_cyclins}b")
            for j, bit in enumerate(binary):
                states_array[i, j] = int(bit)

        return states_array

    def set_starting_states(self, starting_states: npt.NDArray) -> None:
        """
        Set the starting states for the cyclins.

        :param npt.NDArray starting_states: A 2D array where each row represents a starting state.
        """
        if not isinstance(starting_states, np.ndarray):
            raise TypeError("Starting states must be a numpy array.")
        if starting_states.ndim != 2:
            raise ValueError("Starting states must be a 2D array.")
        self.__start_states = starting_states

    def filter_starting_states(self, zero_cyclins: list, one_cyclins: list) -> None:
        """
        Filter the starting states keeping certain nodes' values fixed to zero or one.
        This method modifies the `__start_states` attribute of the class.
        First get the indices of the cyclins that should be in state 0 or 1,
        then filter the starting states based on these indices.

        :param list zero_cyclins: List of cyclins that should be in state 0.
        :param list one_cyclins: List of cyclins that should be in state 1.
        """
        zero_ixs = [self.__get_cyclin_index(cyclin) for cyclin in zero_cyclins]
        one_ixs = [self.__get_cyclin_index(cyclin) for cyclin in one_cyclins]
        # For numpy arrays, use boolean masking
        if isinstance(self.__start_states, np.ndarray):
            # Create masks for the zero and one conditions
            zero_mask = np.all(self.__start_states[:, zero_ixs] == 0, axis=1)
            one_mask = np.all(self.__start_states[:, one_ixs] == 1, axis=1)
            # Combine masks and filter states
            combined_mask = zero_mask & one_mask
            self.__start_states = self.__start_states[combined_mask]
        else:
            # Fallback for list case
            self.__start_states = [
                state
                for state in self.__start_states
                if all(state[ix] == 0 for ix in zero_ixs) and all(state[ix] == 1 for ix in one_ixs)
            ]

    @partial(jit, static_argnums=(0,))
    def _async_calculate_next_state(self, current_state: npt.NDArray, cyclin_ix: int) -> npt.NDArray:
        """
        Calculate the next state based on the current state of the cyclins.

        :param npt.NDArray current_state: A 2D array where each row represents the current state of the cyclins.
        :returns: npt.NDArray: The next state of the cyclins.
        """
        # batch_size, n_cyclins = current_state.shape

        # state_values = jnp.matmul(current_state)

        # Start with a copy of the current state
        next_state = current_state.copy()

        # Get the row for the specific cyclin (incoming edges to this cyclin)
        cyclin_row = graph_matrix[cyclin_ix]

        # Set self-loop to zero to exclude it from calculation
        cyclin_row_no_self = cyclin_row.at[cyclin_ix].set(0)

        # Calculate state value using dot product: sum(edge_weight * source_state)
        # Shape: (batch_size,) = (batch_size, n_cyclins) @ (n_cyclins,)
        state_values = jnp.dot(current_state, cyclin_row_no_self)

        # Apply Boolean logic for the specific cyclin
        new_cyclin_states = jnp.where(state_values > 0, 1, jnp.where(state_values < 0, 0, current_state[:, cyclin_ix]))

        # Apply self-loop rules if no change occurred
        if self.__self_activation_flag or self.__self_inhibition_flag:
            no_change_mask = new_cyclin_states == current_state[:, cyclin_ix]
            new_cyclin_states = self._apply_async_self_loops(
                graph_matrix, current_state, new_cyclin_states, cyclin_ix, no_change_mask
            )

        # Update only the specific cyclin in the next state
        next_state = next_state.at[:, cyclin_ix].set(new_cyclin_states)

        return next_state

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
        # Shape: (batch_size, n_cyclins) = (batch_size, n_cyclins) @ (n_cyclins, n_cyclins)
        state_values = jnp.matmul(current_state, graph_no_self.T)

        # Apply Boolean logic for all cyclins
        next_state = jnp.where(state_values > 0, 1, jnp.where(state_values < 0, 0, current_state))

        # Apply self-loop rules for positions with no change
        if self.__self_activation_flag or self.__self_inhibition_flag:
            no_change_mask = next_state == current_state
            next_state = self._apply_sync_self_loops(graph_matrix, current_state, next_state, no_change_mask)

        return next_state

    @partial(jit, static_argnums=(0,))
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
        # Get incoming edges excluding self-loop
        incoming_edges = jnp.concatenate(
            [graph_matrix[cyclin_ix, :cyclin_ix], graph_matrix[cyclin_ix, cyclin_ix + 1 :]]
        )

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
    def _apply_sync_self_loops(
        self,
        graph_matrix: jnp.ndarray,
        current_state: jnp.ndarray,
        next_state: jnp.ndarray,
        no_change_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply self-loop rules for synchronous updates on all cyclins.

        :param jnp.ndarray graph_matrix: The interaction graph matrix
        :param jnp.ndarray current_state: Current state of all cyclins
        :param jnp.ndarray next_state: Calculated next states for all cyclins
        :param jnp.ndarray no_change_mask: Boolean mask for positions with no change
        :return jnp.ndarray: Updated states with self-loop rules applied
        """
        n_cyclins = graph_matrix.shape[0]

        # Apply self-loop rules for each cyclin
        for cyclin_ix in range(n_cyclins):
            # Get incoming edges excluding self-loop
            incoming_edges = jnp.concatenate(
                [graph_matrix[cyclin_ix, :cyclin_ix], graph_matrix[cyclin_ix, cyclin_ix + 1 :]]
            )

            red_count = jnp.sum(incoming_edges == -1)
            green_count = jnp.sum(incoming_edges == 1)

            # Apply self-inhibition rule
            if self.__self_inhibition_flag:
                inhibition_condition = (red_count == 0) | (green_count > red_count)
                apply_inhibition = no_change_mask[:, cyclin_ix] & inhibition_condition
                next_state = next_state.at[:, cyclin_ix].set(jnp.where(apply_inhibition, 0, next_state[:, cyclin_ix]))

            # Apply self-activation rule
            if self.__self_activation_flag:
                activation_condition = (green_count == 0) | (red_count > green_count)
                apply_activation = no_change_mask[:, cyclin_ix] & activation_condition
                next_state = next_state.at[:, cyclin_ix].set(jnp.where(apply_activation, 1, next_state[:, cyclin_ix]))

        return next_state
