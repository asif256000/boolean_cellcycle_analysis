import random
from copy import deepcopy

# from inputs import all_final_states_to_ignore, expected_cyclin_order, g1_state_zero_cyclins
from logs_inner import logger

# from mammal_inputs import all_final_states_to_ignore, expected_cyclin_order, g1_state_zero_cyclins


class CellCycleStateCalculation:
    def __init__(
        self, cyclins: set | list, expected_final_state: dict, g1_states_only: bool = False, organism: str = "yeast"
    ) -> None:
        """Initialization function for the CellCycleStateCalculation class. This function sets and initializes few required class variables.

        :param set | list cyclins: Cyclins that are selected to play a role in the Cell cycle that are to be calculated.
        :param dict expected_final_state: The expected state that is taken as the control state to help calculate score for different graphs and starting states.
        :param bool g1_states_only: If only G1 start states are required to be taken into consideration, set to True. defaults to False.
        :param str organism: This flag helps identify which organism we are calculating states for, so that we use proper parameters. defaults to yeast.
        """
        if organism.lower() == "yeast":
            self.__init_yeast_specific_vars()
        else:
            self.__init_mammal_specific_vars()
        self.all_cyclins = cyclins
        self.cyclin_print_map = {f"P{ix:>02}": c for ix, c in enumerate(cyclins)}

        self.start_states = self.__get_all_possible_starting_states()
        if g1_states_only:
            self.start_states = self.__get_all_g1_states()

        self.set_expected_final_state(expected_final_state)

    def __init_yeast_specific_vars(self):
        from inputs import all_final_states_to_ignore, expected_cyclin_order, g1_state_zero_cyclins

        self.all_final_states_to_ignore = all_final_states_to_ignore
        self.expected_cyclin_order = expected_cyclin_order
        self.g1_state_zero_cyclins = g1_state_zero_cyclins
        self.optimal_graph_score = 751
        self.optimal_g1_graph_score = 2111
        self.self_activation_flag = False
        self.self_deactivation_flag = True

    def __init_mammal_specific_vars(self):
        from mammal import all_final_states_to_ignore, expected_cyclin_order, g1_state_zero_cyclins

        self.all_final_states_to_ignore = all_final_states_to_ignore
        self.expected_cyclin_order = expected_cyclin_order
        self.g1_state_zero_cyclins = g1_state_zero_cyclins
        self.optimal_graph_score = 4171
        self.optimal_g1_graph_score = 2111
        self.self_activation_flag = True
        self.self_deactivation_flag = True

    def __get_all_possible_starting_states(self) -> list[dict]:
        """Generates all possible starting states from the list of cyclins that are set as the class variable.

        :return list[dict]: For example, [{"C1": 0, "C2": 1,...}, ...] for self.all_cyclins = ["C1", "C2", ...]
        """
        num_of_cyclins = len(self.all_cyclins)
        binary_states_list = [f"{i:>0{num_of_cyclins}b}" for i in range(2**num_of_cyclins)]

        return [dict(zip(self.all_cyclins, map(int, list(state)))) for state in binary_states_list]

    def __get_all_g1_states(self) -> list[dict]:
        """From already calculated self.start_states, filters out the ones for which some given cyclins (taken as input: g1_state_zero_cyclins) are non-zero.

        :return list[dict]: Same as __get_all_possible_starting_states function.
        """
        return [
            state for state in self.start_states if all(state[cyclin] == 0 for cyclin in self.g1_state_zero_cyclins)
        ]

    def set_starting_state(self, starting_states: list):
        """Set the starting state from the input parameter.

        :param list starting_states: List of states. Similar to ones from the function __get_all_possible_starting_states.
        """
        self.start_states = starting_states

    def set_expected_final_state(self, expected_final_state: dict):
        """Set the expected final state from the input parameter.

        :param dict expected_final_state: Expected final state which is considered a control state for calculating graph score.
        """
        self.expected_final_state = expected_final_state

    def set_green_full_connected_graph(self) -> str:
        """Generate a graph where all cyclins are connected to each other with positive (green) edge.

        :return str: Return "All Connected" harcoded string as modification identifier for fully connected graph.
        """
        if isinstance(self.all_cyclins, set):
            self.nodes_and_edges = {cyclin: {1: self.all_cyclins.difference([cyclin])} for cyclin in self.all_cyclins}
        elif isinstance(self.all_cyclins, list):
            self.nodes_and_edges = dict()
            for cyclin in self.all_cyclins:
                self.nodes_and_edges[cyclin] = {1: [c for c in self.all_cyclins if c != cyclin]}
        self.graph_modification = "All Connected"
        return self.graph_modification

    def set_custom_connected_graph(self, graph: dict[dict]) -> str:
        """Set custom graph taken as input parameter to the class variable for graph to be iterated over.

        :param dict[dict] graph: Graph that is to be set.
        :return str: Hardcoded string "Custom Graph" as modification identifier.
        """
        self.nodes_and_edges = graph
        self.graph_modification = "Custom Graph"
        return self.graph_modification

    def set_random_modified_graph(self, original_graph: dict[dict]) -> str:
        """Randomly modify two edges from the graph taken as input parameter and set the modified graph to the class variable.

        :param dict[dict] original_graph: Graph, represented by a dictionary of dictionary.
        Every key of the first level is individual cyclins, and the keys in the next level of dict is 0, 1 and -1
        representing no connection, positive connection and negative connection respectively.
        Values corresponding to the keys should combine to be all_cyclins class variable.
        :return str: Identifier of the random changes, comma separated.
        Pattern: "For node=<target_node>: move <source_node> from <from_random_key> to <to_random_key>, For node=..."
        """
        change_tracker = list()
        try:
            # Get two cyclins randomly from the graph
            two_random_nodes = set(random.choices(tuple(original_graph.keys()), k=2))
        except IndexError as ie:
            logger.error(f"Graph has no node. {original_graph=}. Error: {ie}")
            two_random_nodes = set()

        for node, incoming_edges in original_graph.items():
            if node in two_random_nodes:
                # Shuffle edges for the randomly selected nodes
                current_change = self.edge_shuffle(incoming_edges)
                change_tracker.append(f"For {node=}: {current_change}")

        self.nodes_and_edges = original_graph

        self.graph_modification = ", ".join(change_tracker)
        return self.graph_modification

    def edge_shuffle(self, edge_map: dict[set]) -> str:
        """For the given dictionary of edges corresponding to a node, shuffle the edges randomly and return a string that identifies the changes done.

        :param dict[set] edge_map: The dictionary of edges for a node in the graph. The keys are always -1, 0, 1 and the union of all the values should be the self.all_cyclins list.
        :return str: Identification of the changes being done randomly for the node.
        """
        weight_list = [-1, 0, 1]
        while True:
            from_edge = random.choice(weight_list)
            if edge_map[from_edge]:
                break
        random_edges = set(random.choices(tuple(edge_map[from_edge]), k=1))
        weight_list.remove(from_edge)
        to_edge = random.choice(weight_list)
        edge_map[from_edge].difference_update(random_edges)
        edge_map[to_edge].update(random_edges)

        return f"move {', '.join(random_edges)} from {from_edge} to {to_edge}"

    def __self_degradation_loop(self, cyclin: str):
        """Checks for specific conditions to decide whether self-degrading loop should be applied to the given node (cyclin).
        If there is no red arrow towards a node, or if number of green arrows are greater than the number of red arrows, and
        if there is no change in the state of the cyclin from the previous state, then the state is turned to zero (0).

        :param str cyclin: The node (cyclin) for which the decision is to be made.
        """
        red_arrow_count = len(self.nodes_and_edges.get(cyclin, dict()).get(-1, set()))
        green_arrow_count = len(self.nodes_and_edges.get(cyclin, dict()).get(1, set()))
        if red_arrow_count == 0 or green_arrow_count > red_arrow_count:
            return True
        return False

    def __self_improvement_loop(self, cyclin: str):
        """Checks for specific conditions to decide whether self-improving loop should be applied to the given node (cyclin).
        If there is no green arrow towards a node, or if number of red arrows are greater than the number of green arrows, and
        if there is no change in the state of the cyclin from the previous state, then the state is turned to one (1).

        :param str cyclin: The node (cyclin) for which the decision is to be made.
        """
        red_arrow_count = len(self.nodes_and_edges.get(cyclin, dict()).get(-1, set()))
        green_arrow_count = len(self.nodes_and_edges.get(cyclin, dict()).get(1, set()))
        if green_arrow_count == 0 or red_arrow_count > green_arrow_count:
            return True
        return False

    def __calculate_next_step(self, current_state: dict) -> dict:
        next_state = dict()

        for cyclin, curr_state in current_state.items():
            summed_value = 0

            for edge_weight, nodes in self.nodes_and_edges.get(cyclin, dict()).items():
                if edge_weight != 0:
                    for node in nodes:
                        summed_value += edge_weight * current_state[node]
            if summed_value > 0:
                next_state[cyclin] = 1
            elif summed_value < 0:
                next_state[cyclin] = 0
            else:
                next_state[cyclin] = curr_state
                if (
                    self.self_deactivation_flag
                    and self.__self_degradation_loop(cyclin)
                    and current_state[cyclin] == next_state[cyclin]
                ):
                    next_state[cyclin] = 0
                if (
                    self.self_activation_flag
                    and self.__self_improvement_loop(cyclin)
                    and current_state[cyclin] == next_state[cyclin]
                ):
                    next_state[cyclin] = 1

        return next_state

    def generate_state_table(
        self, starting_state: dict, iteration_count: int, verify_state_sequence: bool = False
    ) -> list[dict]:
        curr_exp_state_order = deepcopy(self.expected_cyclin_order)
        cyclin_states = list()
        cyclin_states.append(starting_state)
        curr_state = starting_state
        if starting_state in self.all_final_states_to_ignore:
            verify_state_sequence = False

        for _ in range(iteration_count):
            if verify_state_sequence and curr_exp_state_order and isinstance(curr_exp_state_order[0], dict):
                if curr_exp_state_order[0].items() <= curr_state.items():
                    found_state = curr_exp_state_order.pop(0)
                    logger.debug(f"Expected State: {found_state!r} found as a subset of Current State: {curr_state!r}")
            curr_state = self.__calculate_next_step(curr_state)
            cyclin_states.append(curr_state)

        if verify_state_sequence and len(curr_exp_state_order) != 0:
            logger.debug("INVALID SEQUENCE OF CYCLIN STATES GENERATED!!!")
            logger.debug(f"State: {curr_exp_state_order[0]} was not found in the generated states.")
            logger.debug(
                f"Graph modification for the iteration: {self.graph_modification} "
                f"Starting state for the iteration: {starting_state}"
            )
            self.print_state_table(cyclin_states, log_level="debug")
            return list()
        return cyclin_states

    def calculate_state_score(self, final_state: dict) -> int:
        score = 0
        for cyclin, exp_state in self.expected_final_state.items():
            score += abs(final_state[cyclin] - exp_state)
        return score

    def iterate_all_states(
        self, view_state_table: bool = False, g1_states_only: bool = False
    ) -> tuple[dict, list[str]]:
        state_scores = dict()
        final_states = list()
        for start_state in self.start_states:
            generated_cyclin_states = self.generate_state_table(
                starting_state=start_state, iteration_count=50, verify_state_sequence=g1_states_only
            )
            if not generated_cyclin_states:
                final_states.append("0" * len(self.all_cyclins))
                state_score = 100
            else:
                final_state = generated_cyclin_states[-1]
                final_states.append("".join(map(str, final_state.values())))
                state_score = self.calculate_state_score(final_state=final_state)
            state_scores["".join(map(str, start_state.values()))] = state_score
            if view_state_table and generated_cyclin_states:
                logger.debug(f"{start_state=}")
                self.print_state_table(generated_cyclin_states)
        return state_scores, final_states

    def calculate_graph_score_and_final_states(
        self, view_state_table: bool = False, view_final_state_count_table: bool = False
    ) -> tuple[int, dict]:
        state_scores, final_states = self.iterate_all_states(view_state_table)

        final_states_count = self.generate_final_state_count_table(final_states)
        if view_final_state_count_table:
            self.print_final_state_count(final_state_count=final_states_count)

        graph_score = sum(state_scores.values())

        logger.debug(f"{graph_score=} for {self.graph_modification=}")
        # if graph_score <= self.optimal_graph_score:
        #     logger.debug(f"GRAPH SCORE LESS THAN OR EQUAL TO {self.optimal_graph_score=}")
        #     logger.debug("Taking a deeper dive into the sequence of states for G1 start states...")
        #     self.start_states = self.__get_all_g1_states()
        #     g1_state_scores, g1_final_states = self.iterate_all_states(view_state_table, g1_states_only=True)
        #     g1_final_state_count = dict()
        #     for g1_fs in set(g1_final_states):
        #         g1_final_state_count[g1_fs] = g1_final_states.count(g1_fs)
        #     g1_state_score_map = {uniq_score: "" for uniq_score in set(g1_state_scores.values())}
        #     for g1_state, state_score in g1_state_scores.items():
        #         g1_state_score_map[state_score] += g1_state
        #         g1_state_score_map[state_score] += ", "
        #     g1_graph_score = sum(g1_state_scores.values())
        #     logger.debug(f"For {self.graph_modification=} {g1_graph_score=}")
        #     if g1_graph_score <= self.optimal_g1_graph_score:
        #         logger.debug(
        #             f"G1 GRAPH SCORE LESS THAN OPTIMAL FOUND!!! {g1_final_state_count=}, {g1_state_score_map=}"
        #         )

        return graph_score, final_states_count

    def generate_final_state_count_table(self, final_states: list[str]) -> dict:
        counter = dict()
        for state in set(final_states):
            counter[state] = final_states.count(state)
        return counter

    def print_final_state_count(self, final_state_count: dict, log_level: str = "debug"):
        table_as_str = f"\n{self.cyclin_print_map}\n"
        table_as_str += "\t|\t".join(["Cnt"] + list(self.cyclin_print_map.keys()))
        table_as_str += "\n"
        for state, count in final_state_count.items():
            table_as_str += f"{count}\t|\t"
            table_as_str += "\t|\t".join(list(state))
            table_as_str += "\n"
        if log_level.lower() == "info":
            logger.info(table_as_str)
        else:
            logger.debug(table_as_str)

    def print_state_table(self, cyclin_states: list[dict], log_level: str = "debug"):
        table_as_str = f"\n{self.cyclin_print_map}\n"
        # headers = list(cyclin_states[0].keys())
        headers = list(self.cyclin_print_map.keys())
        table_as_str += "\t|\t".join(["Tm"] + headers)
        table_as_str += "\n"
        for i in range(len(cyclin_states)):
            table_as_str += f"{i + 1}\t|\t"
            table_as_str += "\t|\t".join([str(cyclin_states[i][self.cyclin_print_map[col]]) for col in headers])
            table_as_str += "\n"
        if log_level.lower() == "info":
            logger.info(table_as_str)
        else:
            logger.debug(table_as_str)
