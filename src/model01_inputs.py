cyclins = ["Cln3", "MBF", "SBF", "Cln1,2", "Cdh1", "Swi5", "Cdc2014", "Clb5,6", "Sic1", "Clb1,2", "Mcm1,SFF"]
modified_graph = [
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "Cln3"
    [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # "MBF"
    [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # "SBF"
    [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # "Cln1,2"
    [0, 0, 0, -1, 0, 0, 1, -1, 0, -1, 0],  # "Cdh1"
    [0, 0, 0, 0, 0, -1, 1, 0, 0, -1, 1],  # "Swi5"
    [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1],  # "Cdc2014"
    [0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0],  # "Clb5,6"
    [0, 0, 0, -1, 0, 1, 1, -1, 0, -1, 0],  # "Sic1"
    [0, 0, 0, 0, -1, 0, -1, 1, -1, 0, 1],  # "Clb1,2"
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -1],  # "Mcm1,SFF"
]

expected_final_state = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
all_final_states_to_ignore = [
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
]

original_graph = [
    [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "Cln3"
    [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # "MBF"
    [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # "SBF"
    [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # "Cln1,2"
    [0, 0, 0, -1, 0, 0, 1, -1, 0, -1, 0],  # "Cdh1"
    [0, 0, 0, 0, 0, -1, 1, 0, 0, -1, 1],  # "Swi5"
    [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1],  # "Cdc2014"
    [0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0],  # "Clb5,6"
    [0, 0, 0, -1, 0, 1, 1, -1, 0, -1, 0],  # "Sic1"
    [0, 0, 0, 0, -1, 0, -1, 1, -1, 0, 1],  # "Clb1,2"
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -1],  # "Mcm1,SFF"
]

custom_start_states = [
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
]

g1_state_zero_cyclins = ["Cdc2014", "Clb5,6", "Cln1,2", "Clb1,2", "Mcm1,SFF"]
g1_state_one_cyclins = ["Cln3", "Cdh1", "Sic1"]

expected_cyclin_order = [
    {"Clb5,6": 1, "Clb1,2": 0, "Cdc2014": 0},
    {"Clb1,2": 1, "Cdc2014": 0},
    {"Cdc2014": 1},
]

# other start states to check than G1 start states
# expected cyclin order is verified for these start states as well
extra_states_to_check = [
    {
        "name": "late_g1_state",
        "start_state_zero_cyclins" : ["Clb5,6", "Cdc2014", "Clb1,2", "Mcm1,SFF"],
        "start_state_one_cyclins" : ["Cln1,2"],
        "expected_cyclin_order" : [
            {"Clb5,6": 1, "Clb1,2": 0, "Cdc2014": 0},
            {"Clb1,2": 1, "Cdc2014": 0},
            {"Cdc2014": 1},
        ]
    },
    {
        "name": "s_state",
        "start_state_zero_cyclins" : ["Cdc2014", "Clb1,2", "Mcm1,SFF"],
        "start_state_one_cyclins" : ["Clb5,6"],
        "expected_cyclin_order" : [
            {"Clb1,2": 1, "Cdc2014": 0},
            {"Cdc2014": 1},
        ]
    },
    {
        "name": "g2/m_state",
        "start_state_zero_cyclins" : ["Cdc2014"],
        "start_state_one_cyclins" : ["Clb1,2"],
        "expected_cyclin_order" : [
            {"Cdc2014": 1},
        ]
    },
]