nodes_and_edges = {
    "Cln3": {1: {}, -1: {}},
    "MBF": {1: {"Cln3"}, -1: {"Clb1,2"}},
    "SBF": {1: {"Cln3"}, -1: {"Clb1,2"}},
    "Cln1,2": {1: {"SBF"}},
    "Clb5,6": {1: {"MBF"}, -1: {"Sic1", "Cdc2014"}},
    "Sic1": {1: {"Swi5", "Cdc2014"}, -1: {"Clb5,6", "Cln1,2", "Clb1,2"}},
    "Clb1,2": {1: {"Mcm1/SFF", "Clb5,6"}, -1: {"Sic1", "Cdc2014", "Cdh1"}},
    "Cdh1": {1: {"Cdc2014"}, -1: {"Clb1,2", "Clb5,6", "Cln1,2"}},
    "Mcm1/SFF": {1: {"Clb1,2", "Clb5,6"}},
    "Cdc2014": {1: {"Clb1,2", "Mcm1/SFF"}},
    "Swi5": {1: {"Cdc2014", "Mcm1/SFF"}, -1: {"Clb1,2"}},
}

starting_state = {
    "Cln3": 1,
    "MBF": 0,
    "SBF": 0,
    "Cln1,2": 0,
    "Cdh1": 1,
    "Swi5": 0,
    "Cdc2014": 0,
    "Clb5,6": 0,
    "Sic1": 1,
    "Clb1,2": 0,
    "Mcm1/SFF": 0,
}

# Harcoding self-degrading cycles not used anymore after implementing generic logic
self_degrading_cyclins = {"Swi5", "Cdc2014", "Mcm1/SFF", "Cln1,2", "Cln3"}


def calculate_next_step(current_state: dict):
    next_state = dict()

    for cyclin, curr_state in current_state.items():
        summed_value = 0

        for edge_weight, nodes in nodes_and_edges.get(cyclin, dict()).items():
            for node in nodes:
                summed_value += edge_weight * current_state[node]

        if summed_value > 0:
            next_state[cyclin] = 1
        elif summed_value < 0:
            next_state[cyclin] = 0
        else:
            next_state[cyclin] = curr_state
            if len(nodes_and_edges.get(cyclin, dict()).get(-1, set())) == 0 or len(
                nodes_and_edges.get(cyclin, dict()).get(1, set())
            ) > len(nodes_and_edges.get(cyclin, dict()).get(-1, set())):
                # if -1 in nodes_and_edges.get(cyclin, dict()).keys():
                # if cyclin in self_degrading_cyclins:
                if current_state[cyclin] == next_state[cyclin]:
                    next_state[cyclin] = 0

    return next_state


def get_all_states():
    cyclin_states = list()
    cyclin_states.append(starting_state)
    curr_state = starting_state

    for _ in range(13):
        curr_state = calculate_next_step(curr_state)
        cyclin_states.append(curr_state)

    return cyclin_states


def print_table(lod: list[dict]):
    for d in lod:
        headers = lod[0].keys()
    temp_head = ["Time"] + list(headers)
    print(*temp_head, sep="\t|\t")
    for i in range(len(lod)):
        print(f"{i+1}\t|\t", end="")
        for col in headers:
            print(lod[i][col], end="\t|\t")
        print("\n")


if __name__ == "__main__":
    all_protien_states = get_all_states()
    print_table(all_protien_states)
    # for elem in all_protien_states:
    #     print(elem)
