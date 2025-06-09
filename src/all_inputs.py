from dataclasses import dataclass, field
from typing import get_origin


# Template of input data that is to be used for all the models
# The placeholder values are to be set to None in the template
@dataclass
class InputTemplate:
    organism: str = None  # The organism for which the model is being tested
    cyclins: list[str] = field(default_factory=list)  # List of cyclins that are present in the model
    new_cyclins: list[str] = field(
        default_factory=list
    )  # List of new cyclins that are to be added to the model and discover new edges
    cell_cycle_activation_cyclin: str = None  # The cyclin that activates the cell cycle
    # The boolean graph of the model, represented as a 2D list with each element representing incoming edge
    # We use this in the simulation, while the original_graph is kept as a backup of the original graph
    modified_graph: list[list[int]] = field(default_factory=list)
    optimal_graph_score: int = None  # The optimal score of the graph
    g1_only_optimal_graph_score: int = None  # The optimal score of the graph when only G1 states are considered
    rule_based_self_activation: bool = None
    rule_based_self_deactivation: bool = None
    # The expected final state of the model, depending on which the final score is calculated.
    # '-' is used instead of 0 and 1 if the state is not important and any value is acceptable.
    expected_final_state: list = field(default_factory=list)
    all_final_states_to_ignore: list[list[int]] = field(default_factory=list)
    original_graph: list[list[int]] = field(default_factory=list)
    custom_start_states: list[list[int]] = field(default_factory=list)
    g1_state_zero_cyclins: list[str] = field(default_factory=list)
    g1_state_one_cyclins: list[str] = field(default_factory=list)
    expected_cyclin_order: list[dict[str, int]] = field(default_factory=list)

    def __post_init__(self):
        for name, field_type in self.__annotations__.items():
            value = self.__dict__[name]

            # Handle parameterized generics (e.g., List[str], Dict[str, int])
            if get_origin(field_type):  # Check if it's a parameterized generic type
                origin = get_origin(field_type)  # Base type (e.g., List)
                if not isinstance(value, origin):  # Check against the base type (e.g., List)
                    current_type = type(value)
                    raise TypeError(f"The field `{name}` was assigned by `{current_type}` instead of `{field_type}`")
            else:
                # Standard isinstance check for non-parameterized types
                if not isinstance(value, field_type):
                    current_type = type(value)
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


# Inputs for Model A (Yeast Model - Model01)
@dataclass
class ModelAInputs(InputTemplate):
    organism: str = field(default="Yeast - Model01")
    cyclins: list[str] = field(
        default_factory=lambda: [
            "Cln3",
            "MBF",
            "SBF",
            "Cln1,2",
            "Cdh1",
            "Swi5",
            "Cdc2014",
            "Clb5,6",
            "Sic1",
            "Clb1,2",
            "Mcm1,SFF",
        ]
    )
    new_cyclins: list[str] = field(default_factory=lambda: ["Cdc2014"])
    cell_cycle_activation_cyclin: str = field(default="Clb5,6")
    optimal_graph_score: int = field(default=751)
    g1_only_optimal_graph_score: int = field(default=2111)
    rule_based_self_activation: bool = field(default=False)
    rule_based_self_deactivation: bool = field(default=True)
    modified_graph: list[list[int]] = field(
        default_factory=lambda: [
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
    )
    expected_final_state: list = field(default_factory=lambda: [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    all_final_states_to_ignore: list[list[int]] = field(
        default_factory=lambda: [
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    original_graph: list[list[int]] = field(
        default_factory=lambda: [
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
    )
    custom_start_states: list[list[int]] = field(
        default_factory=lambda: [
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    g1_state_zero_cyclins: list[str] = field(
        default_factory=lambda: ["Cdc2014", "Clb5,6", "Cln1,2", "Clb1,2", "Mcm1,SFF"]
    )
    g1_state_one_cyclins: list[str] = field(default_factory=lambda: ["Cln3", "Cdh1", "Sic1"])
    expected_cyclin_order: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"Clb5,6": 1, "Clb1,2": 0, "Cdc2014": 0},
            {"Clb1,2": 1, "Cdc2014": 0},
            {"Cdc2014": 1},
        ]
    )


# Inputs for Model B (Mammal Model derived from modifying Faure boolean Model - Model02)
@dataclass
class ModelBInputs(InputTemplate):
    organism: str = field(default="Mammal - Model02")
    cyclins: list[str] = field(
        default_factory=lambda: ["CycD", "CycE", "CycA", "Cdc20", "CycB", "E2F", "RB", "P27", "Cdh1", "Cdc14", "XX"]
    )
    new_cyclins: list[str] = field(default_factory=lambda: ["Cdc20", "XX"])
    cell_cycle_activation_cyclin: str = "CycE"
    optimal_graph_score: int = field(default=4171)
    g1_only_optimal_graph_score: int = field(default=2111)
    rule_based_self_activation: bool = field(default=True)
    rule_based_self_deactivation: bool = field(default=True)
    g1_state_zero_cyclins: list[str] = field(default_factory=lambda: ["CycE", "CycA", "CycB", "Cdc20"])
    g1_state_one_cyclins: list[str] = field(default_factory=lambda: ["CycD", "RB", "P27", "Cdh1"])

    expected_final_state: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 1, "-", 1, 0, "-"])

    expected_cyclin_order: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"E2F": 1, "CycE": 0, "CycA": 0, "CycB": 0, "Cdc20": 0},
            {"CycE": 1, "CycA": 0, "CycB": 0, "Cdc20": 0},
            {"CycA": 1, "CycB": 0, "Cdc20": 0},
            {"CycB": 1, "Cdc20": 0},
            {"Cdc20": 1},
        ]
    )

    all_final_states_to_ignore: list[list[int]] = field(default_factory=lambda: [])

    modified_graph: list[list[int]] = field(
        default_factory=lambda: [
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],  # "CycD"
            [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0],  # "CycE"
            [0, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0],  # "CycA"
            [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],  # "Cdc20"
            [0, 0, 0, -1, 1, 0, 0, 0, -1, 0, 0],  # "CycB"
            [0, 0, -1, 0, -1, 1, -1, 0, 0, 0, 0],  # "E2F"
            [-1, -1, -1, 0, -1, 0, 1, 0, 0, 1, 0],  # "RB"
            [-1, -1, -1, 0, -1, 0, 0, -1, 0, 1, 0],  # "P27"
            [0, -1, -1, 0, -1, 0, 0, 0, 0, 1, 0],  # "Cdh1"
            [0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0],  # "Cdc14"
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "XX"
        ]
    )

    original_graph: list[list[int]] = field(
        default_factory=lambda: [
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
    )

    custom_start_states: list[list[int]] = field(
        default_factory=lambda: [
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 1],
        ]
    )


# Inputs for Model C (Mammal Model derived from modifying Goldbeter probabilistic Model - Model03)
@dataclass
class ModelCInputs(InputTemplate):
    organism: str = field(default="Mammal - Model03")
    cyclins: list[str] = field(
        default_factory=lambda: [
            "CycD",
            "CycE",
            "CycA",
            "CycB",
            "E2F",
            "Skp2",
            "Cdh1",
            "Cdc25",
            "RB",
            "P21",
            "Cdc20",
            "Wee1",
            "Cdc14",
            "XX",
        ]
    )
    new_cyclins: list[str] = field(default_factory=lambda: ["XX"])
    cell_cycle_activation_cyclin: str = field(default="CycE")
    optimal_graph_score: int = field(default=4171)
    g1_only_optimal_graph_score: int = field(default=2111)
    rule_based_self_activation: bool = field(default=True)
    rule_based_self_deactivation: bool = field(default=True)
    g1_state_zero_cyclins: list[str] = field(default_factory=lambda: ["CycE", "CycA", "CycB", "E2F", "Cdc20"])
    g1_state_one_cyclins: list[str] = field(default_factory=lambda: ["CycD", "RB", "Wee1"])

    expected_final_state: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 1, 0, 1, "-", 0, 1, 0, "-"])

    expected_cyclin_order: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"E2F": 1, "CycE": 0, "CycA": 0, "CycB": 0, "Cdc20": 0},
            {"CycE": 1, "CycA": 0, "CycB": 0, "Cdc20": 0},
            {"CycA": 1, "CycB": 0, "Cdc20": 0},
            {"CycB": 1, "Cdc20": 0},
            {"Cdc20": 1},
        ]
    )

    all_final_states_to_ignore: list[list[int]] = field(default_factory=lambda: [])

    original_graph: list[list[int]] = field(
        default_factory=lambda: [
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
    )
    modified_graph: list[list[int]] = field(
        default_factory=lambda: [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],  # "CycD"
            [0, 0, 0, 0, 1, -1, 0, 0, -1, -1, 0, 0, 0, 0],  # "CycE"
            [0, 0, 0, 0, 1, 0, 0, 1, -1, -1, -1, -1, 0, 0],  # "CycA"
            [0, 0, 0, 1, 0, 0, -1, 1, 0, -1, -1, -1, 0, 0],  # "CycB"
            [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0],  # "E2F"
            [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # "Skp2"
            [0, 0, -1, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # "Cdh1"
            [0, 0, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # "Cdc25"
            [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # "RB"
            [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],  # "P21"
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],  # "Cdc20"
            [0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # "Wee1"
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0],  # "Cdc14"
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # "XX"
        ]
    )

    custom_start_states: list[list[int]] = field(
        default_factory=lambda: [
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        ]
    )
