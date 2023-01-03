from log_module import logger


class CellCycleStateCalculation:
    def __init__(self, organism: str = "yeast", g1_states_only: bool = False) -> None:
        if organism.lower() == "yeast":
            self.__init_yeast_specific_vars()
        else:
            self.__init_mammal_specific_vars()

        self.cyclin_print_map = {f"P{ix:>02}": c for ix, c in enumerate(self.__all_cyclins)}

        self.__start_states = self.__get_all_possible_starting_states()
        self.__g1_states_only_flag = g1_states_only
        self.__g1_start_states = self.__get_all_g1_states()

    def __init_yeast_specific_vars(self):
        from yeast_inputs import (
            all_final_states_to_ignore,
            cyclins,
            expected_cyclin_order,
            expected_final_state,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
        )

        self.__all_cyclins = cyclins
        self.__expected_final_state = expected_final_state
        self.__all_final_states_to_ignore = all_final_states_to_ignore
        self.__expected_cyclin_order = expected_cyclin_order
        self.__g1_state_zero_cyclins = g1_state_zero_cyclins
        self.__g1_state_one_cyclins = g1_state_one_cyclins
        self.__optimal_graph_score = 751
        self.__optimal_g1_graph_score = 2111
        self.__self_activation_flag = False
        self.__self_deactivation_flag = True

    def __init_mammal_specific_vars(self):
        from mammal_inputs import (
            all_final_states_to_ignore,
            cyclins,
            expected_cyclin_order,
            expected_final_state,
            g1_state_one_cyclins,
            g1_state_zero_cyclins,
        )

        self.__all_cyclins = cyclins
        self.__expected_final_state = expected_final_state
        self.__all_final_states_to_ignore = all_final_states_to_ignore
        self.__expected_cyclin_order = expected_cyclin_order
        self.__g1_state_zero_cyclins = g1_state_zero_cyclins
        self.__g1_state_one_cyclins = g1_state_one_cyclins
        self.__optimal_graph_score = 4171
        self.__optimal_g1_graph_score = 2111
        self.__self_activation_flag = True
        self.__self_deactivation_flag = True

    def __get_all_possible_starting_states(self) -> list[list]:
        """ "Generates all possible starting states from the list of cyclins that are set as the class variable.

        :return set[list]: A set of starting states, where their order corresponds to the order of self.all_cyclins.
        """
        num_of_cyclins = len(self.__all_cyclins)
        binary_states_list = [f"{i:>0{num_of_cyclins}b}" for i in range(2**num_of_cyclins)]

        return [list(map(int, list(state))) for state in binary_states_list]

    def __get_all_g1_states(self) -> list:
        g1_start_states = list()
        g1_zero_ixs = [self.__get_cyclin_index(zero_cyclin) for zero_cyclin in self.__g1_state_zero_cyclins]
        g1_one_ixs = [self.__get_cyclin_index(one_cyclin) for one_cyclin in self.__g1_state_one_cyclins]
        for state in self.__start_states:
            if all(state[g1_zero_index] == 0 for g1_zero_index in g1_zero_ixs) and all(
                state[g1_one_index] == 1 for g1_one_index in g1_one_ixs
            ):
                g1_start_states.append(state)
        return g1_start_states

    def __get_cyclin_index(self, cyclin: str):
        return self.__all_cyclins.index(cyclin)

    def set_starting_state(self, starting_states: set):
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
        self.graph_modification = graph_identifier

    def set_random_modified_graph(self):
        ...

    def edge_shuffle(self):
        ...

    def __self_degradation_loop(self, cyclin_index: int) -> bool:
        """Checks for specific conditions to decide whether self-degrading loop should be applied to the given node (cyclin).
        If there is no red arrow towards a node, or if number of green arrows are greater than the number of red arrows, and
        if there is no change in the state of the cyclin from the previous state, then the state is turned to zero (0).

        :param int cyclin_index: The index of the node (cyclin) in the original list of nodes for which the decision is to be made.
        :return bool: True if self degradation is applicable, False otherwise.
        """
        red_arrow_count = self.nodes_and_edges[cyclin_index].count(-1)
        green_arrow_count = self.nodes_and_edges[cyclin_index].count(1)
        if red_arrow_count == 0 or green_arrow_count > red_arrow_count:
            return True
        return False

    def __self_improvement_loop(self, cyclin_index: int) -> bool:
        """Checks for specific conditions to decide whether self-improving loop should be applied to the given node (cyclin).
        If there is no green arrow towards a node, or if number of red arrows are greater than the number of green arrows, and
        if there is no change in the state of the cyclin from the previous state, then the state is turned to one (1).

        :param int cyclin_index: The index of the node (cyclin) in the original list of nodes for which the decision is to be made.
        :return bool: True if self improvement is applicable, False otherwise.
        """
        green_arrow_count = self.nodes_and_edges[cyclin_index].count(1)
        red_arrow_count = self.nodes_and_edges[cyclin_index].count(-1)
        if green_arrow_count == 0 or red_arrow_count > green_arrow_count:
            return True
        return False

    def __decide_self_loops(self, current_state: list, next_state: list, cyclin_ix: int):
        if (
            self.__self_deactivation_flag
            and self.__self_degradation_loop(cyclin_index=cyclin_ix)
            and current_state == next_state[cyclin_ix]
        ):
            next_state[cyclin_ix] = 0
        if (
            self.__self_activation_flag
            and self.__self_improvement_loop(cyclin_index=cyclin_ix)
            and current_state == next_state[cyclin_ix]
        ):
            next_state[cyclin_ix] = 1

    def __calculate_next_step(self, current_state: list) -> list[int]:
        next_state = list()

        for ix, cyclin_state in enumerate(current_state):
            state_value = 0

            for edge_val in self.nodes_and_edges[ix]:
                state_value += edge_val * cyclin_state

            if state_value > 0:
                next_state.append(1)
            elif state_value < 0:
                next_state.append(0)
            else:
                next_state.append(cyclin_state)
                self.__decide_self_loops(cyclin_state, next_state, ix)

        return next_state

    def generate_state_table(self, starting_state: list, iteration_count: int) -> list[list]:
        cyclin_states = [starting_state]
        curr_state = starting_state
        # TODO: Verify State Seq false if start state in ignore list
        for _ in range(iteration_count):
            curr_state = self.__calculate_next_step(current_state=curr_state)
            cyclin_states.append(curr_state)

        return cyclin_states

    def calculate_state_scores(self, final_state: list) -> int:
        score = 0
        for ix, exp_state in enumerate(self.__expected_final_state):
            score += abs(final_state[ix] - exp_state)
        return score

    def iterate_all_start_states(self, view_state_table: bool = False, g1_states_only: bool = False):
        state_scores = dict()
        final_states = list()

        if g1_states_only:
            all_start_states = self.__g1_start_states
        else:
            all_start_states = self.__start_states

        for start_state in all_start_states:
            all_cyclin_states = self.generate_state_table(starting_state=start_state, iteration_count=51)
            curr_final_state = all_cyclin_states[-1]
            final_states.append("".join(map(str, curr_final_state)))
            state_score = self.calculate_state_scores(curr_final_state)
            state_scores["".join(map(str, start_state))] = state_score
            if view_state_table:
                logger.debug(f"{start_state=}")
                self.print_state_table(all_cyclin_states)
        return state_scores, final_states

    @staticmethod
    def generate_final_state_counts(final_states: list) -> dict:
        counter = dict()
        for state in set(final_states):
            counter[state] = final_states.count(state)
        return counter

    def generate_graph_score_and_final_states(
        self, view_state_table: bool = False, view_final_state_count_table: bool = False
    ):
        state_scores, final_states = self.iterate_all_start_states(view_state_table=view_state_table)
        logger.debug(f"{final_states=}")
        logger.debug(f"{set(final_states)}")
        final_states_count = self.generate_final_state_counts(final_states)
        logger.debug(f"{final_states_count=}")
        graph_score = sum(state_scores.values())
        logger.debug(f"{graph_score=} for graph modification={self.graph_modification}")
        if view_final_state_count_table:
            self.print_final_state_count_table(final_state_count=final_states_count)

        if graph_score < self.__optimal_graph_score:
            logger.info(f"Graph score {graph_score} less than original score of {self.__optimal_graph_score}")
            logger.debug("Taking a deeper dive into the sequence of states for G1 start states...")
            g1_state_scores, g1_final_states = self.iterate_all_start_states(g1_states_only=True)
            g1_final_state_count = self.generate_final_state_counts(g1_final_states)
            g1_graph_score = sum(g1_state_scores.values())
            logger.debug(f"For {self.graph_modification=}, {g1_graph_score=}")
            if g1_graph_score <= self.__optimal_g1_graph_score:
                logger.info(
                    f"G1 Graph Score {g1_graph_score} less than original score of {self.__optimal_g1_graph_score}"
                )
                logger.debug(f"Print this dictionary as a table: {g1_final_state_count=}")

        return graph_score, final_states_count

    def print_final_state_count_table(self, final_state_count: dict, log_level: str = "debug"):
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

    def print_state_table(self, cyclin_states: list, log_level: str = "debug"):
        table_as_str = f"\n{self.cyclin_print_map}\n"
        table_as_str += "\t|\t".join(["Tm"] + list(self.cyclin_print_map))
        table_as_str += "\t|\n"
        for ix, ix_state in enumerate(cyclin_states):
            table_as_str += f"{ix+1}\t|\t"
            table_as_str += "\t|\t".join(map(str, ix_state))
            table_as_str += "\t|\n"

        if log_level.lower() == "info":
            logger.info(table_as_str)
        else:
            logger.debug(table_as_str)
