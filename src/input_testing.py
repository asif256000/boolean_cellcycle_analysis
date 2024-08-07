cyclins = ["CycD", "CycE", "CycA", "Cdc20", "CycB", "E2F", "RB", "P27", "Cdh1", "Cdc14"]

original_graph = [
    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, -1, 0, 0],
    [0, 0, 0, -1, 0, 1, 0, -1, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0, -1, 0],
    [0, 0, -1, 0, -1, 1, -1, 0, 0, 0],
    [-1, -1, -1, 1, -1, 0, 1, 0, 0, 1],
    # [-1, -1, -1, 1, 0, 0, 0, 1, 0, 1],
    [-1, -1, -1, 1, 0, 0, 0, 0, 0, 1],
    [0, -1, -1, 1, -1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, -1],
]

modified_graph = [
    [0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, -1, 0, 0],
    [0, 0, 0, -1, 0, 1, 0, -1, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0, 0, -1, 0],
    [0, 0, -1, 0, -1, 1, -1, 0, 0, 0],
    [-1, -1, -1, 1, -1, 0, 1, 0, 0, 1],
    # [-1, -1, -1, 1, 0, 0, 0, 1, 0, 1],
    [-1, -1, -1, 1, 0, 0, 0, 0, 0, 1],
    [0, -1, -1, 1, -1, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, -1],
]
