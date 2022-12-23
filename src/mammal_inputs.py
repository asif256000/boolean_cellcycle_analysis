cyclins = ["CyclinD", "CyclinE", "CyclinA", "CyclinB", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "P21-27", "Cdc20", "Wee1"]
g1_state_zero_cyclins = []
expected_final_state = {
    "CyclinD": 1,
    "CyclinE": 1,
    "CyclinA": 1,
    "CyclinB": 1,
    "E2F1": 0,
    "Skp2": 1,
    "Cdh1": 0,
    "Cdc25": 1,
    "RB": 0,
    "P21-27": 0,
    "Cdc20": 1,
    "Wee1": 0,
}

expected_cyclin_order = []

all_final_states_to_ignore = []

# Structure of the graph:
# Outermost keys are the nodes, and corresponding values are the edges.
# Inner dictionary contains 3 keys: -1, 0, 1, denoting negative influence (red arrow),
# no influence (no arrow) and positive influence (green arrow) respectively.
# The values corresponding to these keys denote the nodes from which
# the +ve, no, or -ve influences are coming to the key node.
# Nb: The union of all 3 values corresponding to the -1, 0, 1 keys
# should contain all the different nodes in the model (`cyclins` list).
# Important: If a value is empty set, it should be written as set(), not {}.
# Python identifies {} as an empty dictionary, not as empty set.
# "CyclinD", "CyclinE", "CyclinA", "CyclinB", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "P21-27", "Cdc20", "Wee1"
original_graph = {
    "CyclinD": {
        1: {"E2F1"},
        -1: {"RB", "P21-27"},
        0: {"CyclinE", "CyclinA", "CyclinB", "Skp2", "Cdh1", "Cdc25", "Cdc20", "Wee1"},
    },
    "CyclinE": {
        1: {"E2F1"},
        -1: {"Skp2", "RB", "P21-27", "Wee1"},
        0: {"CyclinD", "CyclinA", "CyclinB", "Cdh1", "Cdc25", "Cdc20"},
    },
    "CyclinA": {
        1: {"E2F1", "Cdc25"},
        -1: {"RB", "P21-27", "Cdc20", "Wee1"},
        0: {"CyclinD", "CyclinE", "CyclinB", "Skp2", "Cdh1"},
    },
    "CyclinB": {
        1: {"Cdc25"},
        -1: {"Cdh1", "P21-27", "Cdc20", "Wee1"},
        0: {"CyclinD", "CyclinE", "CyclinA", "E2F1", "Skp2", "RB"},
    },
    "E2F1": {
        1: set(),
        -1: {"RB", "CyclinA"},
        0: {"CyclinD", "CyclinE", "CyclinB", "Skp2", "Cdh1", "Cdc25", "P21-27", "Cdc20", "Wee1"},
    },
    "Skp2": {
        1: set(),
        -1: {"Cdh1"},
        0: {"CyclinD", "CyclinE", "CyclinA", "CyclinB", "E2F1", "Cdc25", "RB", "P21-27", "Cdc20", "Wee1"},
    },
    "Cdh1": {
        1: set(),
        -1: {"CyclinA", "CyclinB"},
        0: {"CyclinD", "CyclinE", "E2F1", "Skp2", "Cdc25", "RB", "P21-27", "Cdc20", "Wee1"},
    },
    "Cdc25": {
        1: {"CyclinA", "CyclinB"},
        -1: set(),
        0: {"CyclinD", "CyclinE", "E2F1", "Skp2", "Cdh1", "RB", "P21-27", "Cdc20", "Wee1"},
    },
    "RB": {
        1: set(),
        -1: {"CyclinD", "CyclinE"},
        0: {"CyclinA", "CyclinB", "E2F1", "Skp2", "Cdh1", "Cdc25", "P21-27", "Cdc20", "Wee1"},
    },
    "P21-27": {
        1: set(),
        -1: {"CyclinE", "Skp2"},
        0: {"CyclinD", "CyclinA", "CyclinB", "E2F1", "Cdh1", "Cdc25", "RB", "Cdc20", "Wee1"},
    },
    "Cdc20": {
        1: {"CyclinB"},
        -1: set(),
        0: {"CyclinD", "CyclinE", "CyclinA", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "P21-27", "Wee1"},
    },
    "Wee1": {
        1: set(),
        -1: {"CyclinB"},
        0: {"CyclinD", "CyclinE", "CyclinA", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "P21-27", "Cdc20"},
    },
}

modified_graph = {
    "CyclinD": {
        1: {"E2F1"},
        -1: {"RB", "P21-27"},
        0: {"CyclinE", "CyclinA", "CyclinB", "Skp2", "Cdh1", "Cdc25", "Cdc20", "Wee1"},
    },
    "CyclinE": {
        1: {"E2F1"},
        -1: {"Skp2", "RB", "P21-27"},
        0: {"CyclinD", "CyclinA", "CyclinB", "Cdh1", "Cdc25", "Cdc20", "Wee1"},
    },
    "CyclinA": {
        1: {"E2F1"},
        -1: {"RB", "P21-27", "Cdc20"},
        0: {"CyclinD", "CyclinE", "CyclinB", "E2F1", "Skp2", "Cdh1", "Cdc25", "Wee1"},
    },
    "CyclinB": {
        1: {"Cdc25"},
        -1: {"Cdh1", "P21-27", "Cdc20", "Wee1"},
        0: {"CyclinD", "CyclinE", "CyclinA", "E2F1", "Skp2", "RB"},
    },
    "E2F1": {
        1: set(),
        -1: {"RB", "CyclinA"},
        0: {"CyclinD", "CyclinE", "CyclinB", "Skp2", "Cdh1", "Cdc25", "P21-27", "Cdc20", "Wee1"},
    },
    "Skp2": {
        1: set(),
        -1: {"Cdh1"},
        0: {"CyclinD", "CyclinE", "CyclinA", "CyclinB", "E2F1", "Cdc25", "RB", "P21-27", "Cdc20", "Wee1"},
    },
    "Cdh1": {
        1: set(),
        -1: {"CyclinA"},
        0: {"CyclinD", "CyclinE", "CyclinB", "E2F1", "Skp2", "Cdc25", "RB", "P21-27", "Cdc20", "Wee1"},
    },
    "Cdc25": {
        1: {"CyclinB"},
        -1: set(),
        0: {"CyclinD", "CyclinE", "CyclinA", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "P21-27", "Cdc20", "Wee1"},
    },
    "RB": {
        1: set(),
        -1: {"CyclinD", "CyclinE"},
        0: {"CyclinA", "CyclinB", "E2F1", "Skp2", "Cdh1", "Cdc25", "P21-27", "Cdc20", "Wee1"},
    },
    "P21-27": {
        1: set(),
        -1: {"CyclinE"},
        0: {"CyclinD", "CyclinA", "CyclinB", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "Cdc20", "Wee1"},
    },
    "Cdc20": {
        1: {"CyclinB"},
        -1: set(),
        0: {"CyclinD", "CyclinE", "CyclinA", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "P21-27", "Wee1"},
    },
    "Wee1": {
        1: set(),
        -1: {"CyclinB"},
        0: {"CyclinD", "CyclinE", "CyclinA", "E2F1", "Skp2", "Cdh1", "Cdc25", "RB", "P21-27", "Cdc20"},
    },
}

custom_start_state = {
    "CyclinD": 0,
    "CyclinE": 0,
    "CyclinA": 0,
    "CyclinB": 0,
    "E2F1": 0,
    "Skp2": 0,
    "Cdh1": 1,
    "Cdc25": 0,
    "RB": 1,
    "P21-27": 1,
    "Cdc20": 0,
    "Wee1": 1,
}
