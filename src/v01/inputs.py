cyclins = ["Cln3", "MBF", "SBF", "Cln1,2", "Cdh1", "Swi5", "Cdc2014", "Clb5,6", "Sic1", "Clb1,2", "Mcm1,SFF"]
g1_state_zero_cyclins = ["Swi5", "Cdc2014", "Clb5,6", "Clb1,2", "Mcm1,SFF"]
expected_final_state = {
    "Cln3": 0,
    "MBF": 0,
    "SBF": 0,
    "Cln1,2": 0,
    "Cdh1": 1,
    "Swi5": 0,
    "Cdc2014": 0,
    "Clb5,6": 0,
    "Sic1": 1,
    "Clb1,2": 0,
    "Mcm1,SFF": 0,
}

expected_cyclin_order = [
    {"Cln1,2": 1, "Clb5,6": 0, "Clb1,2": 0, "Cdc2014": 0},
    {"Clb5,6": 1, "Clb1,2": 0, "Cdc2014": 0},
    {"Clb1,2": 1, "Cdc2014": 0},
    {"Cdc2014": 1},
]

all_final_states_to_ignore = [
    {
        "Cln3": 0,
        "MBF": 0,
        "SBF": 0,
        "Cln1,2": 0,
        "Cdh1": 1,
        "Swi5": 0,
        "Cdc2014": 0,
        "Clb5,6": 0,
        "Sic1": 1,
        "Clb1,2": 0,
        "Mcm1,SFF": 0,
    },
    {
        "Cln3": 0,
        "MBF": 0,
        "SBF": 1,
        "Cln1,2": 1,
        "Cdh1": 0,
        "Swi5": 0,
        "Cdc2014": 0,
        "Clb5,6": 0,
        "Sic1": 0,
        "Clb1,2": 0,
        "Mcm1,SFF": 0,
    },
    {
        "Cln3": 0,
        "MBF": 1,
        "SBF": 0,
        "Cln1,2": 0,
        "Cdh1": 1,
        "Swi5": 0,
        "Cdc2014": 0,
        "Clb5,6": 0,
        "Sic1": 1,
        "Clb1,2": 0,
        "Mcm1,SFF": 0,
    },
    {
        "Cln3": 0,
        "MBF": 0,
        "SBF": 0,
        "Cln1,2": 0,
        "Cdh1": 0,
        "Swi5": 0,
        "Cdc2014": 0,
        "Clb5,6": 0,
        "Sic1": 1,
        "Clb1,2": 0,
        "Mcm1,SFF": 0,
    },
    {
        "Cln3": 0,
        "MBF": 1,
        "SBF": 0,
        "Cln1,2": 0,
        "Cdh1": 0,
        "Swi5": 0,
        "Cdc2014": 0,
        "Clb5,6": 0,
        "Sic1": 1,
        "Clb1,2": 0,
        "Mcm1,SFF": 0,
    },
    {
        "Cln3": 0,
        "MBF": 0,
        "SBF": 0,
        "Cln1,2": 0,
        "Cdh1": 0,
        "Swi5": 0,
        "Cdc2014": 0,
        "Clb5,6": 0,
        "Sic1": 0,
        "Clb1,2": 0,
        "Mcm1,SFF": 0,
    },
    {
        "Cln3": 0,
        "MBF": 0,
        "SBF": 0,
        "Cln1,2": 0,
        "Cdh1": 1,
        "Swi5": 0,
        "Cdc2014": 0,
        "Clb5,6": 0,
        "Sic1": 0,
        "Clb1,2": 0,
        "Mcm1,SFF": 0,
    },
]

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
# M = [
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
#     1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0;
#     1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0;%
#     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;%
#     0, 0, 0, -1, 0, 0, 1, -1, 0, -1, 0;%
#     0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 1;%
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1;%
#     0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0;%
#     0, 0, 0, -1, 0, 1, 1, -1, 0, -1, 0;%
#     0, 0, 0, 0, -1, 0, -1, 1, -1, 0, 1;%
#     0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0;
# ];
original_graph = {
    "Cln3": {
        1: set(),
        -1: set(),
        0: {"Cdh1", "MBF", "Clb5,6", "Clb1,2", "Swi5", "Mcm1,SFF", "SBF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "MBF": {
        1: {"Cln3"},
        -1: {"Clb1,2"},
        0: {"Cdh1", "Clb5,6", "Swi5", "Mcm1,SFF", "SBF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "SBF": {
        1: {"Cln3"},
        -1: {"Clb1,2"},
        0: {"Cdh1", "MBF", "Clb5,6", "Swi5", "Mcm1,SFF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "Cln1,2": {
        1: {"SBF"},
        -1: set(),
        0: {"Cdh1", "MBF", "Cln3", "Clb5,6", "Clb1,2", "Swi5", "Mcm1,SFF", "Sic1", "Cdc2014"},
    },
    "Clb5,6": {
        1: {"MBF"},
        -1: {"Sic1", "Cdc2014"},
        0: {"Cdh1", "Cln3", "Clb1,2", "Swi5", "Mcm1,SFF", "SBF", "Cln1,2"},
    },
    "Sic1": {
        1: {"Swi5", "Cdc2014"},
        -1: {"Clb5,6", "Cln1,2", "Clb1,2"},
        0: {"Cdh1", "MBF", "Cln3", "Mcm1,SFF", "SBF"},
    },
    "Clb1,2": {
        1: {"Mcm1,SFF", "Clb5,6"},
        -1: {"Sic1", "Cdc2014", "Cdh1"},
        0: {"MBF", "Cln3", "Swi5", "SBF", "Cln1,2"},
    },
    "Cdh1": {
        1: {"Cdc2014"},
        -1: {"Clb1,2", "Clb5,6", "Cln1,2"},
        0: {"MBF", "Cln3", "Swi5", "Mcm1,SFF", "SBF", "Sic1"},
    },
    "Mcm1,SFF": {
        1: {"Clb1,2", "Clb5,6"},
        -1: set(),
        0: {"Cdh1", "MBF", "Cln3", "Swi5", "SBF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "Cdc2014": {
        1: {"Clb1,2", "Mcm1,SFF"},
        -1: set(),
        0: {"Cdh1", "MBF", "Cln3", "Clb5,6", "Swi5", "SBF", "Sic1", "Cln1,2"},
    },
    "Swi5": {
        1: {"Mcm1,SFF", "Cdc2014"},
        -1: {"Clb1,2"},
        0: {"Cdh1", "MBF", "Cln3", "Clb5,6", "SBF", "Sic1", "Cln1,2"},
    },
}

modified_graph = {
    "Cln3": {
        1: set(),
        -1: set(),
        0: {"Cdh1", "MBF", "Clb5,6", "Clb1,2", "Swi5", "Mcm1,SFF", "SBF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "MBF": {
        1: {"Cln3"},
        -1: {"Clb1,2"},
        0: {"Cdh1", "Clb5,6", "Swi5", "Mcm1,SFF", "SBF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "SBF": {
        1: {"Cln3"},
        -1: {"Clb1,2"},
        0: {"Cdh1", "MBF", "Clb5,6", "Swi5", "Mcm1,SFF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "Cln1,2": {
        1: {"SBF"},
        -1: set(),
        0: {"Cdh1", "MBF", "Cln3", "Clb5,6", "Clb1,2", "Swi5", "Mcm1,SFF", "Sic1", "Cdc2014"},
    },
    "Clb5,6": {
        1: {"MBF"},
        -1: {"Sic1", "Cdc2014"},
        0: {"Cdh1", "Cln3", "Clb1,2", "Swi5", "Mcm1,SFF", "SBF", "Cln1,2"},
    },
    "Sic1": {
        1: {"Swi5", "Cdc2014"},
        -1: {"Clb5,6", "Cln1,2", "Clb1,2"},
        0: {"Cdh1", "MBF", "Cln3", "Mcm1,SFF", "SBF"},
    },
    "Clb1,2": {
        1: {"Mcm1,SFF", "Clb5,6"},
        -1: {"Sic1", "Cdc2014", "Cdh1"},
        0: {"MBF", "Cln3", "Swi5", "SBF", "Cln1,2"},
    },
    "Cdh1": {
        1: {"Cdc2014"},
        -1: {"Clb1,2", "Clb5,6", "Cln1,2"},
        0: {"MBF", "Cln3", "Swi5", "Mcm1,SFF", "SBF", "Sic1"},
    },
    "Mcm1,SFF": {
        1: {"Clb1,2", "Clb5,6"},
        -1: set(),
        0: {"Cdh1", "MBF", "Cln3", "Swi5", "SBF", "Sic1", "Cln1,2", "Cdc2014"},
    },
    "Cdc2014": {
        1: {"Mcm1,SFF"},  # Mod1
        -1: set(),
        0: {"Cdh1", "Clb1,2", "MBF", "Cln3", "Clb5,6", "Swi5", "SBF", "Sic1", "Cln1,2"},
    },
    "Swi5": {
        1: {"Clb1,2", "Mcm1,SFF", "Cdc2014"},
        -1: set(),  # Mod2
        0: {"Cdh1", "MBF", "Cln3", "Clb5,6", "SBF", "Sic1", "Cln1,2"},
    },
}

custom_start_state = {
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
    "Mcm1,SFF": 0,
}