cyclins = ["Cln3", "MBF", "SBF", "Cln1,2", "Cdh1", "Swi5", "Cdc2014", "Clb5,6", "Sic1", "Clb1,2", "Mcm1,SFF"]

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


custom_graph = {
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