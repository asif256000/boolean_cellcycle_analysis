class CellCycleStateCalculation:
    def __init__(self, cyclins: set) -> None:
        self.all_cyclins = cyclins

    def get_all_possible_starting_states(self) -> list[dict]:
        num_of_cyclins = len(self.all_cyclins)
        binary_states_list = [f"{i:>0{num_of_cyclins}b}" for i in range(2**num_of_cyclins)]

        all_start_states = list()
        for state in binary_states_list:
            all_start_states.append(dict(zip(self.all_cyclins, map(int, list(state)))))

        return all_start_states

    def set_green_full_connected_graph(self):
        self.nodes_and_edges = {cyclin: {1: self.all_cyclins - {cyclin}} for cyclin in self.all_cyclins}

    def set_custom_connected_graph(self, graph: dict[dict]):
        self.nodes_and_edges = graph

    def set_starting_state(self, starting_state: dict):
        self.starting_state = starting_state

    def __calculate_next_step(self, current_state: dict) -> dict:
        next_state = dict()

        for cyclin, curr_state in current_state.items():
            summed_value = 0

            for edge_weight, nodes in self.nodes_and_edges.get(cyclin, dict()).items():
                for node in nodes:
                    # print(f"{current_state=}, {cyclin=}, {edge_weight=}, {node=}")
                    summed_value += edge_weight * current_state[node]

            if summed_value > 0:
                next_state[cyclin] = 1
            elif summed_value < 0:
                next_state[cyclin] = 0
            else:
                next_state[cyclin] = curr_state
                # Generic method to determine which nodes have self-degrading loop
                if len(self.nodes_and_edges.get(cyclin, dict()).get(-1, set())) == 0 or len(
                    self.nodes_and_edges.get(cyclin, dict()).get(1, set())
                ) > len(self.nodes_and_edges.get(cyclin, dict()).get(-1, set())):
                    # if -1 in nodes_and_edges.get(cyclin, dict()).keys():
                    # if cyclin in self_degrading_cyclins:
                    if current_state[cyclin] == next_state[cyclin]:
                        next_state[cyclin] = 0

        return next_state

    def generate_state_table(self, iteration_count: int):
        self.cyclin_states = list()
        self.cyclin_states.append(self.starting_state)
        curr_state = self.starting_state

        for _ in range(iteration_count):
            curr_state = self.__calculate_next_step(curr_state)
            self.cyclin_states.append(curr_state)

    def calculate_score(self, expected_state: dict):
        score = 0
        for cyclin, exp_state in expected_state.items():
            score += abs(self.cyclin_states[-1][cyclin] - exp_state)
        return score

    def print_table(self):
        for d in self.cyclin_states:
            headers = self.cyclin_states[0].keys()
        temp_head = ["Time"] + list(headers)
        print(*temp_head, sep="\t|\t")
        for i in range(len(self.cyclin_states)):
            print(f"{i+1}\t|\t", end="")
            for col in headers:
                print(self.cyclin_states[i][col], end="\t|\t")
            print("\n")
