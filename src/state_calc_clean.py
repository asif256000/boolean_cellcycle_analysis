from copy import deepcopy
from random import choice, choices, shuffle

from log_module import logger


class CellCycleStateCalculation:
    def __init__(self, input_json: dict) -> None:
        self.__all_cyclins = input_json["cyclins"]
        self.__organism = input_json["organism"]
        if self.__organism == "yeast":
            self.__init_yeast_specific_vars()
        elif self.__organism == "gb_mammal":
            self.__init_gb_mammal_specific_vars()
        elif self.__organism == "fr_mammal":
            self.__init_fr_mammal_specific_vars()

        self.cyclin_print_map = {f"P{ix:>02}": c for ix, c in enumerate(self.__all_cyclins)}

        self.__detailed_logs = input_json["detailed_logs"]
        self.__hardcoded_self_loops = input_json["hardcoded_self_loops"]
        self.__check_sequence = input_json["check_sequence"]
        self.__g1_states_only_flag = input_json["g1_states_only"]
        self.__view_state_table = input_json["view_state_table"]
        self.__view_state_change_only = input_json["view_state_changes_only"]
        self.__view_final_state_count_table = input_json["view_final_state_count_table"]
        self.__async_update = input_json["async_update"]
        self.__random_order_cyclin = input_json["random_order_cyclin"]
        self.__complete_cycle = input_json["complete_cycle"]
        self.__exp_cycle_detection = input_json["expensive_state_cycle_detection"]
        self.__cell_cycle_activation_cyclin = input_json["cell_cycle_activation_cyclin"]

        logger.set_ignore_details_flag(flag=not self.__detailed_logs)
        self.__start_states = self.__get_all_possible_starting_states()
        self.__g1_start_states = self.__get_all_g1_states()

        logger.debug(f"Class state: {self}")
        logger.debug(f"Inputs: {input_json}")

    def __init_yeast_specific_vars(self):
        from yeast_inputs import (
            all_final_states_to_ignore,
            expected_cyclin_order,
            expected_final_state,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
        )

        self.__expected_final_state = expected_final_state
        self.__all_final_states_to_ignore = all_final_states_to_ignore
        self.__expected_cyclin_order = expected_cyclin_order
        self.__g1_state_zero_cyclins = g1_state_zero_cyclins
        self.__g1_state_one_cyclins = g1_state_one_cyclins
        self.__optimal_graph_score = 751
        self.__optimal_g1_graph_score = 2111
        self.__self_activation_flag = False
        self.__self_deactivation_flag = True

    def __init_gb_mammal_specific_vars(self):
        from gb_mammal_inputs import (
            all_final_states_to_ignore,
            expected_cyclin_order,
            expected_final_state,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
        )

        self.__expected_final_state = expected_final_state
        self.__all_final_states_to_ignore = all_final_states_to_ignore
        self.__expected_cyclin_order = expected_cyclin_order
        self.__g1_state_zero_cyclins = g1_state_zero_cyclins
        self.__g1_state_one_cyclins = g1_state_one_cyclins
        self.__optimal_graph_score = 4171
        self.__optimal_g1_graph_score = 2111
        self.__self_activation_flag = True
        self.__self_deactivation_flag = True

    def __init_fr_mammal_specific_vars(self):
        from fr_mammal_inputs import (
            all_final_states_to_ignore,
            expected_cyclin_order,
            expected_final_state,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
        )

        self.__expected_final_state = expected_final_state
        self.__all_final_states_to_ignore = all_final_states_to_ignore
        self.__expected_cyclin_order = expected_cyclin_order
        self.__g1_state_zero_cyclins = g1_state_zero_cyclins
        self.__g1_state_one_cyclins = g1_state_one_cyclins
        self.__optimal_graph_score = 4171
        self.__optimal_g1_graph_score = 2111
        self.__self_activation_flag = True
        self.__self_deactivation_flag = True

    def __repr__(self) -> str:
        return (
            f"Class CellCycleStateCalculation. Organism: {self.__organism} "
            f"Cyclins: {self.__all_cyclins}, "
            f"Optimal Graph Score: {self.__optimal_graph_score}, "
            f"Expected Final State: {dict(zip(self.__all_cyclins, self.__expected_final_state))}"
        )

    def __get_all_possible_starting_states(self) -> list[list]:
        """ "Generates all possible starting states from the list of cyclins that are set as the class variable.

        :return set[list]: A set of starting states, where their order corresponds to the order of self.all_cyclins.
        """
        num_of_cyclins = len(self.__all_cyclins)
        binary_states_list = [f"{i:>0{num_of_cyclins}b}" for i in range(2**num_of_cyclins)]

        return [list(map(int, list(state))) for state in binary_states_list]

    def __get_cyclin_index(self, cyclin: str) -> int:
        return self.__all_cyclins.index(cyclin)

    def get_optimal_scores(self) -> tuple[int, int]:
        return self.__optimal_graph_score, self.__optimal_g1_graph_score

    def __get_all_g1_states(self) -> list:
        return self.filter_start_states(
            zero_cyclins=self.__g1_state_zero_cyclins, one_cyclins=self.__g1_state_one_cyclins
        )

    def filter_start_states(self, zero_cyclins: list = list(), one_cyclins: list = list()):
        filtered_start_states = list()
        zero_ixs = [self.__get_cyclin_index(zero_cyclin) for zero_cyclin in zero_cyclins]
        one_ixs = [self.__get_cyclin_index(one_cyclin) for one_cyclin in one_cyclins]
        for state in self.__start_states:
            if all(state[zero_index] == 0 for zero_index in zero_ixs) and all(
                state[one_index] == 1 for one_index in one_ixs
            ):
                filtered_start_states.append(state)
        return filtered_start_states

    def set_starting_state(self, starting_states: list):
        for start_state in starting_states:
            if len(start_state) != len(self.__all_cyclins):
                raise Exception(
                    f"Starting State {start_state} length does not match Cyclin {self.__all_cyclins} Length!"
                )
        self.__start_states = starting_states
        for start_state in starting_states:
            if len(start_state) != len(self.__all_cyclins):
                raise Exception(
                    f"Starting State {start_state} length does not match Cyclin {self.__all_cyclins} Length!"
                )
        self.__start_states = starting_states

    def set_expected_final_state(self, final_state: list):
        if len(final_state) != len(self.__all_cyclins):
            raise Exception(f"Final State {final_state} length does not match Cyclin {self.__all_cyclins} Length!")
        self.__expected_final_state = final_state

    def set_custom_connected_graph(self, graph: list[list], graph_identifier: str = "Custom"):
        for ix, edges in enumerate(graph):
            if len(edges) != len(self.__all_cyclins):
                raise Exception(
                    f"Edges {edges} length does not match Cyclins {self.__all_cyclins} length for node number {ix+1}"
                )
        self.nodes_and_edges = graph
        self.graph_copy = [row[:] for row in graph]
        logger.debug(f"Set {graph=} for {graph_identifier=}", detail=True)
        self.graph_modification = graph_identifier

    def set_random_modified_graph(self, og_graph: list[list], change_count: int = 2) -> str:
        graph = deepcopy(og_graph)
        cyclin_len = len(self.__all_cyclins)

        change_tracker = list()
        for _ in range(change_count):
            change_tracker.append(self.__edge_shuffle(cyclin_len=cyclin_len, graph_to_modify=graph))

        self.nodes_and_edges = graph
        self.graph_copy = [row[:] for row in graph]

        return ", ".join(change_tracker)

    def __edge_shuffle(self, cyclin_len: int, graph_to_modify: list) -> str:
        possible_edge_weights = {-1, 0, 1}
        change = choices(range(cyclin_len), k=2)
        x, y = change[0], change[-1]
        from_val = graph_to_modify[x][y]
        to_val = choices(possible_edge_weights - {from_val})
        graph_to_modify[x][y] = to_val

        return f"Pos {x=} and {y=} changed {from_val=} {to_val=}"

    def state_as_str(self, state: list) -> str:
        state_dict = dict(zip(self.__all_cyclins, state))
        return ", ".join([f"{k}: {v}" for k, v in state_dict.items()])

    def __calculate_state_scores(self, final_state: list) -> int:
        score = 0
        for ix, exp_state in enumerate(self.__expected_final_state):
            if isinstance(exp_state, int):
                score += abs(final_state[ix] - exp_state)
        return score

    @staticmethod
    def __generate_final_state_counts(final_states: list) -> dict:
        return {state: final_states.count(state) for state in set(final_states)}

    def __self_degradation_loop(self, graph_matrix: list[list], cyclin_index: int) -> bool:
        """Checks for specific conditions to decide whether self-degrading loop should be applied to the given node (cyclin).
        If there is no red arrow towards a node, or if number of green arrows are greater than the number of red arrows, and
        if there is no change in the state of the cyclin from the previous state, then the state is turned to zero (0).

        :param int cyclin_index: The index of the node (cyclin) in the original list of nodes for which the decision is to be made.
        :return bool: True if self degradation is applicable, False otherwise.
        """
        if self.__hardcoded_self_loops:
            if graph_matrix[cyclin_index][cyclin_index] == -1:
                return True
            else:
                return False
        reqd_list = graph_matrix[cyclin_index][:cyclin_index] + graph_matrix[cyclin_index][cyclin_index + 1 :]
        red_arrow_count = reqd_list.count(-1)
        green_arrow_count = reqd_list.count(1)
        if red_arrow_count == 0 or green_arrow_count > red_arrow_count:
            return True
        return False

    def __self_activating_loop(self, graph_matrix: list[list], cyclin_index: int) -> bool:
        """Checks for specific conditions to decide whether self-improving loop should be applied to the given node (cyclin).
        If there is no green arrow towards a node, or if number of red arrows are greater than the number of green arrows, and
        if there is no change in the state of the cyclin from the previous state, then the state is turned to one (1).

        :param list[list] graph_matrix: Interaction graph represented as a matrix (list of list of int).
        :param int cyclin_index: The index of the node (cyclin) in the original list of nodes for which the decision is to be made.
        :return bool: True if self improvement is applicable, False otherwise.
        """
        if self.__hardcoded_self_loops:
            if graph_matrix[cyclin_index][cyclin_index] == 1:
                return True
            else:
                return False
        reqd_list = graph_matrix[cyclin_index][:cyclin_index] + graph_matrix[cyclin_index][cyclin_index + 1 :]
        green_arrow_count = reqd_list.count(1)
        # red_arrow_count = reqd_list.count(-1)
        if green_arrow_count == 0:  # or red_arrow_count > green_arrow_count:
            return True
        return False

    def __decide_self_loops(self, graph_matrix: list[list], current_state: list, next_state: list, cyclin_ix: int):
        """This function decides whether to apply self-loops (whether degrading or activating) for a particular cyclin in the graph.

        :param list[list] graph_matrix: Interaction graph represented as a matrix (list of list of int).
        :param list current_state: _description_
        :param list next_state: _description_
        :param int cyclin_ix: _description_
        """
        if (
            self.__self_deactivation_flag
            and self.__self_degradation_loop(graph_matrix=graph_matrix, cyclin_index=cyclin_ix)
            and current_state[cyclin_ix] == next_state[cyclin_ix]
        ):
            next_state[cyclin_ix] = 0
        if (
            self.__self_activation_flag
            and self.__self_activating_loop(graph_matrix=graph_matrix, cyclin_index=cyclin_ix)
            and current_state[cyclin_ix] == next_state[cyclin_ix]
        ):
            next_state[cyclin_ix] = 1

    def __async_calculate_next_step(self, graph_matrix: list[list], current_state: list, cyclin_ix: int) -> list:
        next_state = [x for x in current_state]

        state_value = 0
        for current_node_index, edge_val in enumerate(graph_matrix[cyclin_ix]):
            if current_node_index != cyclin_ix:
                state_value += edge_val * current_state[current_node_index]

        if state_value > 0:
            next_state[cyclin_ix] = 1
        elif state_value < 0:
            next_state[cyclin_ix] = 0
        else:
            next_state[cyclin_ix] = current_state[cyclin_ix]
            self.__decide_self_loops(graph_matrix, current_state, next_state, cyclin_ix)

        return next_state

    def __sync_calculate_next_step(self, graph_matrix: list[list], current_state: list) -> list:
        next_state = [x for x in current_state]

        for ix, cyclin_state in enumerate(current_state):
            state_value = 0
            for current_ix, edge_val in enumerate(graph_matrix[ix]):
                if current_ix != ix:
                    state_value += edge_val * current_state[current_ix]

            if state_value > 0:
                next_state[ix] = 1
            elif state_value < 0:
                next_state[ix] = 0
            else:
                next_state[ix] = cyclin_state
                self.__decide_self_loops(graph_matrix, current_state, next_state, ix)

        return next_state

    def __generate_state_table(
        self, graph_matrix: list[list], graph_mod_id: str, start_state: list, iter_count: int
    ) -> list:
        cyclin_states = [start_state]
        curr_state = [x for x in start_state]
        cyclin_count = len(self.__all_cyclins)

        if self.__async_update:
            iter_count = iter_count * cyclin_count

        update_order = list()
        for i in range(iter_count):
            if self.__async_update and self.__random_order_cyclin:
                if self.__complete_cycle:
                    if i % cyclin_count == 0:
                        shuffled_ixs = list(range(cyclin_count))
                        shuffle(shuffled_ixs)
                    c_ix = shuffled_ixs[i % cyclin_count]
                else:
                    c_ix = choice(range(cyclin_count))
                update_order.append(self.__all_cyclins[c_ix])
                curr_state = self.__async_calculate_next_step(
                    graph_matrix=graph_matrix, current_state=curr_state, cyclin_ix=c_ix
                )
            elif self.__async_update and not self.__random_order_cyclin:
                c_ix = i % cyclin_count
                curr_state = self.__async_calculate_next_step(
                    graph_matrix=graph_matrix, current_state=curr_state, cyclin_ix=c_ix
                )
            else:
                curr_state = self.__sync_calculate_next_step(graph_matrix=graph_matrix, current_state=curr_state)
            cyclin_states.append(curr_state)

        if self.__async_update and self.__random_order_cyclin:
            logger.debug(f"Order of random async updates:\n{update_order}", detail=True)

        return cyclin_states, update_order

    @staticmethod
    def remove_continuous_duplicates(all_states: list) -> list:
        return [v for i, v in enumerate(all_states) if i == 0 or v != all_states[i - 1]]

    def __lazy_detect_cycles(self, all_states: list) -> bool:
        if not self.__async_update and all_states[-1] != all_states[-2]:
            return True
        if self.__async_update and all_states[-1] != all_states[-1 * len(self.__all_cyclins)]:
            return True
        return False

    def __detect_any_cycles(self, all_states: list) -> bool:
        """Should return True iff there is an ongoing cycle in the second half of the list. Cycle here can be defined as a
        repeated sequence that appears in the list in the same order multiple times consecutively.

        :param list all_states: The list in which cycle has to be detected.
        :return bool: Returns True iff a cycle is detected in the second half of the list, False otherwise.
        """
        filtered_states = self.remove_continuous_duplicates(all_states)
        states_len = len(filtered_states)
        reverse_states = list(reversed(filtered_states))
        for rev_ix, _ in enumerate(reverse_states[: states_len // 2 + 1]):
            for i in range(2, states_len // 2):
                if reverse_states[rev_ix : rev_ix + i] == reverse_states[rev_ix + i : rev_ix + 2 * i]:
                    return True
        return False

    def __detect_end_cycles(self, all_states: list) -> bool:
        """Should return True iff there is an ongoing cycle at the end of the list. Cycle here can be defined as a repeated sequence
        that appears in the list in the same order multiple times consecutively.

        :param list all_states: The list of states in which cycle has to be detected.
        :return bool: Returns True iff a cycle is detected at the end of the list, False otherwise.
        """
        filtered_states = self.remove_continuous_duplicates(all_states)
        return any(
            filtered_states[-i:] == filtered_states[-2 * i : -i] for i in range(2, len(filtered_states) // 2 + 1)
        )

    def __check_activation_index(self, all_states: list) -> int:
        default_ix = -1
        target_cyclin_ix = self.__get_cyclin_index(self.__cell_cycle_activation_cyclin)
        filtered_states = self.remove_continuous_duplicates(all_states)
        if filtered_states[0][target_cyclin_ix] == 1:
            return default_ix
        for ix, state in enumerate(filtered_states):
            if state[target_cyclin_ix] == 1:
                return ix
        return default_ix

    def __iterate_all_start_states(self, graph_matrix: list[list], graph_mod_id: str) -> tuple[dict, list]:
        state_scores_dict = dict()
        final_states = list()
        incorrect_seq_tracker = list()
        correct_seq_tracker = list()
        not_started_seq_tracker = list()
        state_seq_type = dict()

        if self.__g1_states_only_flag:
            all_start_states = self.__g1_start_states
        else:
            all_start_states = self.__start_states

        for start_state in all_start_states:
            cell_div_start_flag = False
            all_cyclin_states, update_sequence = self.__generate_state_table(
                graph_matrix=graph_matrix, graph_mod_id=graph_mod_id, start_state=start_state, iter_count=100
            )
            if self.__check_activation_index(all_cyclin_states) != -1:
                cell_div_start_flag = True

            if self.__exp_cycle_detection and self.__detect_end_cycles(all_cyclin_states):
                final_states.append("C" * len(self.__all_cyclins))
                state_score = self.__calculate_state_scores(all_cyclin_states[-1]) + self.__calculate_state_scores(
                    all_cyclin_states[-2]
                )
                logger.debug(
                    f"Cycle found for start state: {str(dict(zip(self.__all_cyclins, start_state)))}", detail=True
                )
            elif not self.__exp_cycle_detection and self.__lazy_detect_cycles(all_cyclin_states):
                final_states.append("C" * len(self.__all_cyclins))
                state_score = self.__calculate_state_scores(all_cyclin_states[-1]) + self.__calculate_state_scores(
                    all_cyclin_states[-2]
                )
            else:
                curr_final_state = all_cyclin_states[-1]
                final_states.append("".join(map(str, curr_final_state)))
                state_score = self.__calculate_state_scores(curr_final_state)

            state_scores_dict["".join(map(str, start_state))] = state_score
            if self.__view_state_table:
                self.print_state_table(all_cyclin_states, update_sequence)

            curr_start_state_str = self.state_as_str(start_state)
            if self.__check_sequence and start_state in self.__g1_start_states:
                if not cell_div_start_flag:
                    not_started_seq_tracker.append(curr_start_state_str)
                    state_seq_type["".join(map(str, start_state))] = "did_not_start"
                    if start_state in self.__g1_start_states:
                        state_score += 100
                        state_scores_dict["".join(map(str, start_state))] = state_score
                    continue
                if self.verify_sequence(all_cyclin_states):
                    correct_seq_tracker.append(curr_start_state_str)
                    state_seq_type["".join(map(str, start_state))] = "correct"
                else:
                    logger.debug("Correct Cyclin order not followed for this start_state", detail=True)
                    incorrect_seq_tracker.append(curr_start_state_str)
                    state_seq_type["".join(map(str, start_state))] = "incorrect"
                    if start_state in self.__g1_start_states:
                        state_score += 100
                        state_scores_dict["".join(map(str, start_state))] = state_score

        # tracked_correct_states = "\n".join(correct_seq_tracker)
        # tracked_incorrect_states = "\n".join(incorrect_seq_tracker)
        # non_started_states = "\n".join(not_started_seq_tracker)
        logs = [
            f"\nA total {len(correct_seq_tracker)} starting states out of {len(all_start_states)} went through correct sequence.",
            f"A total {len(incorrect_seq_tracker)} starting states out of {len(all_start_states)} did not go through correct sequence.",
            f"A total {len(not_started_seq_tracker)} starting states out of {len(all_start_states)} did not start cell cycle",
            f"i.e it did not turn or started with {self.__cell_cycle_activation_cyclin} as 1 in the cell cycle.",
            # f"The states that followed correct order are:\n{tracked_correct_states}",
            # f"The states that did not follow correct order are:\n{tracked_incorrect_states}",
            # f"The states that did not start cell cycle are:\n{non_started_states}",
        ]
        logger.debug("\n".join(logs), detail=True)

        return state_scores_dict, final_states, state_seq_type

    def verify_sequence(self, all_states: list) -> bool:
        """Checks if the state path that was traversed follows expected order. Returns True if it follows correct sequence, otherwise returns False.

        :param list all_states: State order that was followed by the cell cycle.
        :return bool: True if expected sequence was followed, otherwise returns False.
        """
        filtered_states = self.remove_continuous_duplicates(all_states)
        expected_order = [order for order in self.__expected_cyclin_order]
        for curr_state in filtered_states:
            if not expected_order:
                break
            if expected_order[0].items() <= dict(zip(self.__all_cyclins, curr_state)).items():
                _ = expected_order.pop(0)

        if len(expected_order) != 0:
            return False
        return True

    def generate_graph_score_and_final_states(
        self, graph_info: tuple[list[list], str] = None
    ) -> tuple[int, dict, dict]:
        logger.set_ignore_details_flag(flag=not self.__detailed_logs)

        if graph_info:
            graph_matrix, graph_mod_id = graph_info
        else:
            graph_matrix, graph_mod_id = self.nodes_and_edges, self.graph_modification

        state_scores, final_states, state_seq_types = self.__iterate_all_start_states(graph_matrix, graph_mod_id)
        final_state_count = self.__generate_final_state_counts(final_states)
        graph_score = sum(state_scores.values())

        logger.debug(f"{graph_score=} for graph modification={graph_mod_id} and organism={self.__organism}")

        if self.__view_final_state_count_table:
            self.print_final_state_count_table(final_state_count)

        return graph_score, final_state_count, state_seq_types

    def print_final_state_count_table_fallback(self, final_state_count: dict, log_level: str = "debug"):
        table_as_str = f"\n{self.cyclin_print_map}\n"
        table_as_str += "\t|\t".join(["Cnt"] + list(self.cyclin_print_map))
        table_as_str += "\t|\n"
        for state, count in final_state_count.items():
            table_as_str += f"{count}\t|\t"
            table_as_str += "\t|\t".join(list(state))
            table_as_str += "\t|\n"

        if log_level.lower() == "info":
            logger.info(table_as_str)
        else:
            logger.debug(table_as_str)

    def print_state_table_fallback(self, cyclin_states: list, random_update_sequence: list, log_level: str = "debug"):
        inv_cyclin_lookup = {v: k for k, v in self.cyclin_print_map.items()}

        curr_tracked_state = cyclin_states[0]
        table_as_str = f"State sequence for start state: {self.state_as_str(curr_tracked_state)}\n"
        table_as_str += f"\n{self.cyclin_print_map}\n"
        table_as_str += "\t|\t".join(["Tm"] + list(self.cyclin_print_map))
        if random_update_sequence:
            table_as_str += "\t|\tUpd\t|\n"
        else:
            table_as_str += "\t|\n"

        for ix, ix_state in enumerate(cyclin_states[1:]):
            if self.__view_state_change_only and ix_state == curr_tracked_state:
                continue
            table_as_str += f"{ix+1}\t|\t"
            table_as_str += "\t|\t".join(map(str, ix_state))
            if random_update_sequence:
                table_as_str += f"\t|\t{inv_cyclin_lookup[random_update_sequence[ix]]}\t|\n"
            else:
                table_as_str += "\t|\n"
            curr_tracked_state = ix_state

        if log_level.lower() == "info":
            logger.info(table_as_str)
        else:
            logger.debug(table_as_str)

    def print_final_state_count_table(self, final_state_count: dict, log_level: str = "debug"):
        table_as_str = "Count of final state for each start states:\n"
        for state, count in final_state_count.items():
            table_as_str += f"Count: {count:>05}, "
            table_as_str += self.state_as_str(state)
            table_as_str += "\n"

        if log_level.lower() == "info":
            logger.info(table_as_str)
        else:
            logger.debug(table_as_str)

    def print_state_table(self, cyclin_states: list, random_update_sequence: list, log_level: str = "debug"):
        curr_tracked_state = cyclin_states[0]
        table_as_str = f"State sequence for start state: {self.state_as_str(curr_tracked_state)}\n"
        table_as_str += "Time: 0000, " + self.state_as_str(curr_tracked_state)
        if random_update_sequence:
            table_as_str += ", Update: Start\n"
        else:
            table_as_str += "\n"

        for ix, ix_state in enumerate(cyclin_states[1:]):
            if self.__view_state_change_only and ix_state == curr_tracked_state:
                continue
            table_as_str += f"Time: {ix+1:>04}, "
            table_as_str += self.state_as_str(ix_state)
            if random_update_sequence:
                table_as_str += f", Update: {random_update_sequence[ix]}\n"
            else:
                table_as_str += "\n"
            curr_tracked_state = ix_state

        if log_level.lower() == "info":
            logger.info(table_as_str)
        else:
            logger.debug(table_as_str)
