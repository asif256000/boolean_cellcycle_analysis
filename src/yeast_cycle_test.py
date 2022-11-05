nodes_and_edges = {
    "MBF": {1: {"Cln3"}, -1: {"Clb1,2"}},
    "SBF": {1: {"Cln3"}, -1: {"Clb1,2"}},
    "Cln1,2": {1: {"SBF"}},
    "Clb5,6": {1: {"MBF"}, -1: {"Sic1", "Cdc20&Cdc14"}},
    "Sic1": {1: {"Swi5", "Cdc20&Cdc14"}, -1: {"Cln3", "Cln1,2", "Clb1,2"}},
    "Clb1,2": {1: {"Mcm1/SFF", "Clb5,6"}, -1: {"Sic1", "Cdc20&Cdc14", "Cdh1"}},
    "Cdh1": {1: {"Cdc20&Cdc14"}, -1: {"Clb1,2", "Clb5,6", "Cln1,2"}},
    "Mcm1/SFF": {1: {"Clb1,2", "Clb5,6"}},
    "Cdc20&Cdc14": {1: {"Clb1,2", "Mcm1/SFF"}},
    "Swi5": {1: {"Cdc20&Cdc14", "Mcm1/SFF"}, -1: {"Clb1,2"}},
}

starting_state = {
    "Cln3": 1,
    "MBF": 0,
    "SBF": 0,
    "Cln1,2": 0,
    "Cdh1": 1,
    "Swi5": 0,
    "Cdc20&Cdc14": 0,
    "Clb5,6": 0,
    "Sic1": 1,
    "Clb1,2": 0,
    "Mcm1/SFF": 0,
}


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

    return next_state


def get_all_states():
    cyclin_states = list()
    cyclin_states.append(starting_state)
    curr_state = starting_state

    for _ in range(15):
        curr_state = calculate_next_step(curr_state)
        cyclin_states.append(curr_state)

    return cyclin_states


if __name__ == "__main__":
    all_protien_states = get_all_states()
    for elem in all_protien_states:
        print(elem)
