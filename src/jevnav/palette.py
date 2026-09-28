from .costmap import GOAL_SYMBOL, ROBOT_SYMBOL, ROUTE_SYMBOL

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
ROUTE = "#2a78d6"
COMMAND = "#eb6834"
GOAL = "#1baf7a"
BAR = "#86b6ef"
BAR_CHOSEN = "#1c5cab"
UNKNOWN_FILL = "#f0efec"
NEAR_FILL = "#d9d7cf"

CELL_COLORS = {
    "?": UNKNOWN_FILL,
    ".": SURFACE,
    "+": NEAR_FILL,
    "x": INK_MUTED,
    "#": INK,
    ROUTE_SYMBOL: ROUTE,
    GOAL_SYMBOL: GOAL,
    ROBOT_SYMBOL: COMMAND,
}


def rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[position : position + 2], 16) for position in (1, 3, 5))
