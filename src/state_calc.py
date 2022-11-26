import random

from inputs import g1_state_zero_cyclins
from log_module import logger


class CellCycleStateCalculation:
    def __init__(self, cyclins: set | list, expected_final_state: dict, g1_states_only: bool = False) -> None:
        self.all_cyclins = cyclins

        self.start_states = self.__get_all_possible_starting_states()
        if g1_states_only:
            self.start_states = self.__get_all_g1_states()

        self.expected_final_state = expected_final_state

    def __get_all_possible_starting_states(self) -> list[dict]:
        num_of_cyclins = len(self.all_cyclins)
        binary_states_list = [f"{i:>0{num_of_cyclins}b}" for i in range(2**num_of_cyclins)]

        return [dict(zip(self.all_cyclins, map(int, list(state)))) for state in binary_states_list]

    def __get_all_g1_states(self) -> list[dict]:
        return [state for state in self.start_states if all(state[cyclin] == 0 for cyclin in g1_state_zero_cyclins)]

    def set_green_full_connected_graph(self):
        if isinstance(self.all_cyclins, set):
            self.nodes_and_edges = {cyclin: {1: self.all_cyclins.difference([cyclin])} for cyclin in self.all_cyclins}
        elif isinstance(self.all_cyclins, list):
            self.nodes_and_edges = dict()
            for cyclin in self.all_cyclins:
                self.nodes_and_edges[cyclin] = {1: [c for c in self.all_cyclins if c != cyclin]}

    def set_custom_connected_graph(self, graph: dict[dict]) -> str:
        self.nodes_and_edges = graph
        return "NA"

    def set_expected_final_state(self, expected_final_state: dict):
        self.expected_final_state = expected_final_state

    def set_starting_state(self, starting_states: list):
        self.start_states = starting_states

    def set_random_modified_graph(self, original_graph: dict[dict]) -> str:
        change_tracker = list()
        try:
            two_random_nodes = set(random.choices(tuple(original_graph.keys()), k=2))
        except IndexError as ie:
            logger.error(f"Graph has no node. {original_graph=}. Error: {ie}")
            two_random_nodes = {}

        for node, incoming_edges in original_graph.items():
            if node in two_random_nodes:
                current_change = self.edge_shuffle(incoming_edges)
                change_tracker.append(f"{node} {current_change}")

        self.nodes_and_edges = original_graph

        return ", ".join(change_tracker)

    def edge_shuffle(self, edge_map: dict[set]) -> str:
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

        return f"from-{from_edge}-{', '.join(random_edges)}-to-{to_edge}"

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
                if len(self.nodes_and_edges.get(cyclin, dict()).get(-1, set())) == 0 or len(
                    self.nodes_and_edges.get(cyclin, dict()).get(1, set())
                ) > len(self.nodes_and_edges.get(cyclin, dict()).get(-1, set())):
                    if current_state[cyclin] == next_state[cyclin]:
                        next_state[cyclin] = 0

        return next_state

    def generate_state_table(self, starting_state: dict, iteration_count: int) -> list[dict]:
        cyclin_states = list()
        cyclin_states.append(starting_state)
        curr_state = starting_state

        for _ in range(iteration_count):
            curr_state = self.__calculate_next_step(curr_state)
            cyclin_states.append(curr_state)

        return cyclin_states

    def calculate_state_score(self, final_state: dict) -> int:
        score = 0
        for cyclin, exp_state in self.expected_final_state.items():
            score += abs(final_state[cyclin] - exp_state)
        return score

    def iterate_all_states(self, view_state_table: bool = False) -> tuple[dict, list]:
        state_scores = dict()
        final_states = list()
        for start_state in self.start_states:
            generated_cyclin_states = self.generate_state_table(starting_state=start_state, iteration_count=13)
            final_state = generated_cyclin_states[-1]
            final_states.append("".join(map(str, final_state.values())))
            state_score = self.calculate_state_score(final_state=final_state)
            state_scores["".join(map(str, start_state.values()))] = state_score
            if view_state_table:
                logger.info(f"{start_state=}")
                self.print_state_table(generated_cyclin_states)
        return state_scores, final_states

    def calculate_graph_score_and_final_states(self, view_state_table: bool = False) -> tuple[int, dict]:
        state_scores, final_states = self.iterate_all_states(view_state_table)
        final_states_count = self.generate_final_state_count_table(final_states)
        return sum(state_scores.values()), final_states_count

    def generate_final_state_count_table(self, final_states: list[dict]) -> dict:
        counter = dict()
        for state in set(final_states):
            counter[state] = final_states.count(state)
        return counter

    def print_final_state_count(self, final_state_count: dict):
        table_as_str = "\n"
        table_as_str += "\t|\t".join(["Count"] + self.all_cyclins)
        table_as_str += "\n"
        for state, count in final_state_count.items():
            table_as_str += f"{count}\t|\t"
            table_as_str += "\t|\t".join(list(state))
            table_as_str += "\n"
        logger.info(table_as_str)

    def print_state_table(self, cyclin_states: list[dict]):
        table_as_str = "\n"
        headers = list(cyclin_states[0].keys())
        table_as_str += "\t|\t".join(["Time"] + headers)
        table_as_str += "\n"
        for i in range(len(cyclin_states)):
            table_as_str += f"{i + 1}\t|\t"
            table_as_str += "\t|\t".join([str(cyclin_states[i][col]) for col in headers])
            table_as_str += "\n"
        logger.info(table_as_str)
