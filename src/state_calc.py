import random


class CellCycleStateCalculation:
    def __init__(self, cyclins: set | list, expected_final_state: dict) -> None:
        self.all_cyclins = cyclins
        self.start_states = self.__get_all_possible_starting_states()
        self.expected_final_state = expected_final_state

    def __get_all_possible_starting_states(self) -> list[dict]:
        num_of_cyclins = len(self.all_cyclins)
        binary_states_list = [f"{i:>0{num_of_cyclins}b}" for i in range(2**num_of_cyclins)]

        all_start_states = list()
        for state in binary_states_list:
            all_start_states.append(dict(zip(self.all_cyclins, map(int, list(state)))))

        return all_start_states

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
            print(f"Graph has no node. {original_graph=}. Error: {ie}")
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

    def calculate_graph_score(self) -> int:
        state_scores = dict()
        for start_state in self.start_states:
            generated_cyclin_states = self.generate_state_table(starting_state=start_state, iteration_count=50)
            state_score = self.calculate_state_score(final_state=generated_cyclin_states[-1])
            state_scores["".join(map(str, start_state.values()))] = state_score
            # self.print_table(generated_cyclin_states)
        return sum(state_scores.values())

    def print_table(self, cyclin_states: list[dict]):
        for d in cyclin_states:
            headers = cyclin_states[0].keys()
        temp_head = ["Time"] + list(headers)
        print(*temp_head, sep="\t|\t")
        for i in range(len(cyclin_states)):
            print(f"{i+1}\t|\t", end="")
            for col in headers:
                print(cyclin_states[i][col], end="\t|\t")
            print("\n")
