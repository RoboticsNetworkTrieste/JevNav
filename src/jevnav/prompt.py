import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .commands import Command
from .costmap import LocalCostmap
from .costmap_image import CELL_PIXELS, png_base64
from .kinematics import wrap_angle
from .route import Route
from .simulation import simulate_commands

PATH_SPACING = 0.4


class EvidenceFormat(StrEnum):
    GRID = "grid"
    TEXT = "text"
    SIMULATION = "simulation"
    IMAGE = "image"


SIMULATED = (EvidenceFormat.SIMULATION, EvidenceFormat.IMAGE)


GRID_HEADER = (
    "Local costmap around a differential-drive robot.\n"
    "The map has {size} rows and {size} columns, {resolution:g} m per cell. R is where the robot "
    "will be when the\nchosen command starts, always the centre cell, facing the top row: row 1 "
    "is {reach:.1f} m ahead,\ncolumn 1 is {reach:.1f} m to the left.\n"
    "Legend: # obstacle, x too close (collision), + near obstacle, . free, ? unknown,\n"
    "* planned route to the goal, G goal."
)

GRID_INSTRUCTIONS = (
    "Choose the motion command the robot will hold for the next {hold:.1f} s. Every move is at "
    "{speed:g} m/s and rotations turn {turn:g} degrees per second. Follow the planned route (*) "
    "toward the goal (G). Never drive into obstacle (#) or too-close (x) cells; avoid "
    "near-obstacle (+) and unknown (?) cells when you can."
)

TEXT_DESCRIPTION = (
    "A differential-drive robot of radius {radius:g} m, described where it will be when the "
    "chosen command starts.\n"
    "Robot: at {position} in the world, heading {heading} degrees.\n"
    "Coordinates below are in meters relative to the robot: x is ahead (negative: behind), "
    "y is to the left (negative: right).\n"
    "Goal: {goal}.\n"
    "Planned path to the goal, a point every {spacing:g} m, nearest first: {path}\n"
    "Obstacles seen by the lidar within {reach:g} m ahead, behind or to either side, "
    "nearest first: {obstacles}"
)

TEXT_INSTRUCTIONS = (
    "Choose the motion command the robot will hold for the next {hold:.1f} s. Every move is at "
    "{speed:g} m/s and rotations turn {turn:g} degrees per second. Follow the planned path "
    "toward the goal. An obstacle point closer than {radius:g} m to the robot centre is a "
    "collision; keep at least {clearance:g} m from every obstacle point when you can."
)

SIMULATION_STATE = (
    "Differential-drive robot of radius {radius:g} m, at the pose where the next command starts. "
    "It moves at {speed:g} m/s or rotates in place at {turn:g} degrees per second. The chosen "
    "command is held for {hold:.1f} s.\n"
    "Every command offered was simulated for the whole hold and is collision-free; commands that "
    "would collide are not offered.\n"
    "Rule: prefer the command that leaves the robot with the least distance still to go along "
    "the planned route to the goal. Among commands that leave a similar distance to go, prefer "
    "the one that ends closer to the route, then the one that ends farther from obstacles. The "
    "straight-line distance to the goal only breaks the remaining ties.\n"
    "Every distance in an option is measured where the robot is when the command ends. Closest "
    "obstacle is the gap between the robot's edge and the nearest obstacle the lidar sees."
)

SIMULATION_INSTRUCTIONS = "Which command should the robot hold for the next {hold:.1f} s?"

IMAGE_CAPTION = (
    "The image is the local costmap from the lidar, centred on the robot at the pose where the "
    "next command starts, facing the top edge. It spans {extent:.1f} m by {extent:.1f} m, "
    "{resolution:g} m per cell; the image's left is the robot's left. Black: obstacle. Dark "
    "gray: too close, the robot would collide. Light gray: near an obstacle. White: free. Pale "
    "gray with a dot: unknown. Blue: the planned route to the goal. Green: the goal. Orange "
    "triangle: the robot."
)

INSTRUCTION_SUFFIX = (
    " Follow this operator instruction, even where it conflicts with the rule or preferences "
    "given: {instruction}"
)

SIMULATION_OPTION = (
    "{motion} After it: {route_to_go:.2f} m to go along the route, {off_route:.2f} m off the "
    "route, {goal_distance:.2f} m straight to the goal; {clearance}."
)


@dataclass(frozen=True)
class DecisionRequest:
    index: int
    sent_step: int
    costmap: LocalCostmap
    evidence: str
    instructions: str
    options: tuple[Command, ...]
    hold: float
    evidence_format: EvidenceFormat = EvidenceFormat.GRID
    descriptions: dict[str, str] = field(default_factory=dict)
    verdicts: dict = field(default_factory=dict)
    instruction: str | None = None
    images: tuple[str, ...] = ()

    def criteria(self) -> dict[str, str]:
        return {
            command.id: self.descriptions.get(command.id, command.text())
            for command in self.options
        }

    def command(self, command_id: str) -> Command:
        for command in self.options:
            if command.id == command_id:
                return command
        raise KeyError(command_id)


def compose(
    index: int,
    sent_step: int,
    costmap: LocalCostmap,
    commands,
    hold: float,
    rng: np.random.Generator,
    evidence_format: EvidenceFormat = EvidenceFormat.GRID,
    *,
    candidates=None,
    route: Route | None = None,
    step: float = 0.1,
    verdicts: dict | None = None,
    instruction: str | None = None,
    image_cell: int = CELL_PIXELS,
) -> DecisionRequest:
    spec = costmap.spec
    forward = next(command for command in commands if command.id == "F")
    rotate = next(command for command in commands if command.id == "L")
    offered = tuple(commands if candidates is None else candidates)
    options = tuple(offered[position] for position in rng.permutation(len(offered)))
    motion = {"hold": hold, "speed": forward.linear, "turn": round(rotate.turn_rate_degrees, 1)}
    descriptions = {}
    images = ()
    if evidence_format in SIMULATED:
        if route is None or costmap.goal_point is None or verdicts is None:
            raise ValueError("Simulation evidence needs the route, the goal and safety verdicts")
        results = simulate_commands(options, costmap.pose, hold, step, route, costmap.goal_point)
        evidence = SIMULATION_STATE.format(radius=spec.robot_radius, **motion)
        instructions = SIMULATION_INSTRUCTIONS.format(hold=hold)
        descriptions = {
            command.id: SIMULATION_OPTION.format(
                motion=command.text(),
                route_to_go=results[command.id].route_to_go,
                off_route=results[command.id].off_route,
                goal_distance=results[command.id].goal_distance,
                clearance=_clearance(verdicts[command.id].end_clearance),
            )
            for command in options
        }
        if evidence_format == EvidenceFormat.IMAGE:
            extent = spec.size * spec.resolution
            evidence += "\n" + IMAGE_CAPTION.format(extent=extent, resolution=spec.resolution)
            images = (png_base64(costmap, image_cell),)
    elif evidence_format == EvidenceFormat.TEXT:
        evidence = describe(costmap)
        instructions = TEXT_INSTRUCTIONS.format(
            **motion, radius=spec.robot_radius, clearance=spec.inflation
        )
    else:
        header = GRID_HEADER.format(
            size=spec.size, resolution=spec.resolution, reach=spec.half_width
        )
        evidence = f"{header}\n\n{costmap.text()}"
        instructions = GRID_INSTRUCTIONS.format(**motion)
    if instruction:
        instructions += INSTRUCTION_SUFFIX.format(instruction=instruction)
    return DecisionRequest(
        index=index,
        sent_step=sent_step,
        costmap=costmap,
        evidence=evidence,
        instructions=instructions,
        options=options,
        hold=hold,
        evidence_format=EvidenceFormat(evidence_format),
        descriptions=descriptions,
        verdicts=dict(verdicts or {}),
        instruction=instruction or None,
        images=images,
    )


def describe(costmap: LocalCostmap) -> str:
    x, y, heading = costmap.pose
    return TEXT_DESCRIPTION.format(
        radius=costmap.spec.robot_radius,
        position=_point((x, y)),
        heading=round(math.degrees(float(wrap_angle(heading)))),
        goal=_goal(costmap),
        spacing=PATH_SPACING,
        path=_points(local_path(costmap)) or "none in view",
        reach=costmap.spec.half_width,
        obstacles=_points(_nearest_first(costmap.local_obstacles())) or "none",
    )


def local_path(costmap: LocalCostmap) -> np.ndarray:
    if len(costmap.route_points) == 0:
        return np.empty((0, 2))
    visible = Route.through(costmap.route_points)
    stations = np.arange(0.0, visible.length + 1e-9, PATH_SPACING)
    if visible.length - stations[-1] > PATH_SPACING / 2:
        stations = np.append(stations, visible.length)
    return costmap.to_local([visible.point_at(station) for station in stations])


def _goal(costmap: LocalCostmap) -> str:
    if costmap.goal_point is None:
        return "not given"
    local = costmap.to_local(costmap.goal_point)[0]
    return f"{_point(local)}, {math.hypot(*local):.1f} m away"


def _nearest_first(points: np.ndarray) -> np.ndarray:
    distance = np.round(np.hypot(points[:, 0], points[:, 1]), 6)
    return points[np.lexsort((points[:, 1], -points[:, 0], distance))]


def _points(points) -> str:
    return " ".join(_point(point) for point in points)


def _point(point) -> str:
    return f"({_meters(point[0])}, {_meters(point[1])})"


def _meters(value) -> str:
    return f"{round(float(value), 1) + 0.0:.1f}"


def _clearance(value: float) -> str:
    if not math.isfinite(value):
        return "no obstacle in lidar range"
    return f"closest obstacle {value:.2f} m"
