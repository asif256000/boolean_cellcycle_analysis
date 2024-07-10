from dataclasses import dataclass


@dataclass
class InputTemplate:
    cyclins: list[str] = None
    modified_graph: list[list[int]] = None
    expected_final_state: list = None
    all_final_states_to_ignore: list[list[int]] = None
    original_graph: list[list[int]] = None
    custom_start_states: list[list[int]] = None
    g1_state_zero_cyclins: list[str] = None
    g1_state_one_cyclins: list[str] = None
    expected_cyclin_order: list[dict[str, int]] = None

    def __post_init__(self):
        for name, field_type in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")

    def __str__(self):
        return (
            f"InputTemplate(cyclins={self.cyclins}, modified_graph={self.modified_graph}, "
            f"expected_final_state={self.expected_final_state}, "
            f"all_final_states_to_ignore={self.all_final_states_to_ignore}, "
            f"original_graph={self.original_graph}, custom_start_states={self.custom_start_states}, "
            f"g1_state_zero_cyclins={self.g1_state_zero_cyclins}, "
            f"g1_state_one_cyclins={self.g1_state_one_cyclins}, "
            f"expected_cyclin_order={self.expected_cyclin_order})"
        )


@dataclass
class ModelAInputs(InputTemplate):
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


@dataclass
class ModelBInputs(InputTemplate):
    cyclins = ["CycD", "CycE", "CycA", "Cdc20", "CycB", "E2F", "RB", "P27", "Cdh1", "Cdc14"]

    g1_state_zero_cyclins = ["CycE", "CycA", "CycB", "Cdc20"]
    g1_state_one_cyclins = ["CycD", "RB", "P27", "Cdh1"]

    expected_final_state = [0, 0, 0, 0, 0, 0, 1, "-", 1, 0]

    expected_cyclin_order = [
        {"E2F": 1, "CycE": 0, "CycA": 0, "CycB": 0, "Cdc20": 0},
        {"CycE": 1, "CycA": 0, "CycB": 0, "Cdc20": 0},
        {"CycA": 1, "CycB": 0, "Cdc20": 0},
        {"CycB": 1, "Cdc20": 0},
        {"Cdc20": 1},
    ]

    all_final_states_to_ignore = []

    modified_graph = [
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # "CycD"
        [0, 0, 0, 0, 0, 1, 0, -1, 0, 0],  # "CycE"
        [0, 0, 0, -1, 0, 1, 0, -1, 0, 0],  # "CycA"
        [0, 0, 0, -1, 1, 0, 0, 0, 0, 0],  # "Cdc20"
        [0, 0, 0, -1, 1, 0, 0, 0, -1, 0],  # "CycB"
        [0, 0, -1, 0, -1, 1, -1, 0, 0, 0],  # "E2F"
        [-1, -1, -1, 0, -1, 0, 1, 0, 0, 1],  # "RB"
        [-1, -1, -1, 0, -1, 0, 0, -1, 0, 1],  # "P27"
        [0, -1, -1, 0, -1, 0, 0, 0, 0, 1],  # "Cdh1"
        [0, 0, 0, 1, 0, 0, 0, 0, 0, -1],  # "Cdc14"
    ]

    original_graph = [
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # "CycD"
        [0, 0, 0, 0, 0, 1, 0, -1, 0, 0],  # "CycE"
        [0, 0, 0, -1, 0, 1, 0, -1, 0, 0],  # "CycA"
        [0, 0, 0, -1, 1, 0, 0, 0, 0, 0],  # "Cdc20"
        [0, 0, 0, -1, 1, 0, 0, 0, -1, 0],  # "CycB"
        [0, 0, -1, 0, -1, 1, -1, 0, 0, 0],  # "E2F"
        [-1, -1, -1, 0, -1, 0, 1, 0, 0, 1],  # "RB"
        [-1, -1, -1, 0, -1, 0, 0, 1, 0, 1],  # "P27"
        [0, -1, -1, 0, -1, 0, 0, 0, 0, 1],  # "Cdh1"
        [0, 0, 0, 1, 0, 0, 0, 0, 0, -1],  # "Cdc14"
    ]

    custom_start_states = [
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        # [0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        # [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        # [1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    ]


@dataclass
class ModelCInputs(InputTemplate):
    cyclins = ["CycD", "CycE", "CycA", "CycB", "E2F", "Skp2", "Cdh1", "Cdc25", "RB", "P21", "Cdc20", "Wee1", "Cdc14"]

    g1_state_zero_cyclins = ["CycE", "CycA", "CycB", "E2F", "Cdc20"]
    g1_state_one_cyclins = ["CycD", "RB", "Wee1"]

    expected_final_state = [0, 0, 0, 0, 0, 0, 1, 0, 1, "-", 0, 1, 0]

    expected_cyclin_order = [
        {"E2F": 1, "CycE": 0, "CycA": 0, "CycB": 0, "Cdc20": 0},
        {"CycE": 1, "CycA": 0, "CycB": 0, "Cdc20": 0},
        {"CycA": 1, "CycB": 0, "Cdc20": 0},
        {"CycB": 1, "Cdc20": 0},
        {"Cdc20": 1},
    ]

    all_final_states_to_ignore = []

    original_graph = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],  # "CycD"
        [0, 0, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0],  # "CycE"
        [0, 0, 0, 0, 1, 0, 0, 1, -1, -1, -1, -1, 0],  # "CycA"
        [0, 0, 0, 1, 0, 0, -1, 1, 0, -1, -1, -1, 0],  # "CycB"
        [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0],  # "E2F"
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # "Skp2"
        [0, 0, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # "Cdh1"
        [0, 0, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # "Cdc25"
        [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # "RB"
        [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],  # "P21"
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # "Cdc20"
        [0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # "Wee1"
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],  # "Cdc14"
    ]
    modified_graph = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],  # "CycD"
        [0, 0, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0],  # "CycE"
        [0, 0, 0, 0, 1, 0, 0, 1, -1, -1, -1, -1, 0],  # "CycA"
        [0, 0, 0, 1, 0, 0, -1, 1, 0, -1, -1, -1, 0],  # "CycB"
        [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0],  # "E2F"
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # "Skp2"
        [0, 0, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # "Cdh1"
        [0, 0, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # "Cdc25"
        [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # "RB"
        [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],  # "P21"
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # "Cdc20"
        [0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # "Wee1"
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],  # "Cdc14"
    ]

    custom_start_states = [
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        # [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
        # [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        # [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        # [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
    ]
