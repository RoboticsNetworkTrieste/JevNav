from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .route import Route

UNKNOWN, FREE, NEAR, TOO_CLOSE, OBSTACLE = range(5)
LAYER_SYMBOLS = {UNKNOWN: "?", FREE: ".", NEAR: "+", TOO_CLOSE: "x", OBSTACLE: "#"}
ROUTE_SYMBOL, GOAL_SYMBOL, ROBOT_SYMBOL = "*", "G", "R"
NO_RETURN_MARGIN = 0.02
RAY_SAMPLES_PER_CELL = 4


@dataclass(frozen=True)
class CostmapSpec:
    size: int = 21
    resolution: float = 0.2
    robot_radius: float = 0.2
    inflation: float = 0.6

    @property
    def center(self) -> int:
        return self.size // 2

    @property
    def half_width(self) -> float:
        return self.center * self.resolution


@dataclass(frozen=True)
class Scan:
    origin: np.ndarray
    ranges: np.ndarray
    angles: np.ndarray
    range_max: float

    @classmethod
    def from_robot(cls, robot) -> "Scan":
        lidar = robot.lidar
        ranges = np.asarray(lidar.range_data, dtype=float).copy()
        return cls(
            origin=np.asarray(lidar.lidar_origin, dtype=float).ravel()[:3].copy(),
            ranges=ranges,
            angles=np.linspace(lidar.angle_min, lidar.angle_max, len(ranges)),
            range_max=float(lidar.range_max),
        )

    @property
    def hits(self) -> np.ndarray:
        return self.ranges < self.range_max - NO_RETURN_MARGIN

    @property
    def points(self) -> np.ndarray:
        bearing = self.angles + self.origin[2]
        return np.column_stack(
            [
                self.origin[0] + self.ranges * np.cos(bearing),
                self.origin[1] + self.ranges * np.sin(bearing),
            ]
        )

    def nearest_hit(self) -> float:
        return float(np.min(self.ranges[self.hits])) if self.hits.any() else float("inf")


@dataclass(frozen=True)
class LocalCostmap:
    spec: CostmapSpec
    pose: np.ndarray
    layers: np.ndarray
    route: np.ndarray
    goal: tuple[int, int] | None
    route_points: np.ndarray
    goal_point: np.ndarray | None = None

    @property
    def target(self) -> np.ndarray | None:
        return self.route_points[-1] if len(self.route_points) else None

    def to_local(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        offset = xy - self.pose[:2]
        cos, sin = np.cos(self.pose[2]), np.sin(self.pose[2])
        forward = offset[:, 0] * cos + offset[:, 1] * sin
        left = -offset[:, 0] * sin + offset[:, 1] * cos
        return np.column_stack([forward, left])

    def to_cells(self, xy) -> np.ndarray:
        return self.spec.center - self.to_local(xy) / self.spec.resolution

    def local_obstacles(self) -> np.ndarray:
        cells = np.argwhere(self.layers == OBSTACLE)
        return (self.spec.center - cells) * self.spec.resolution

    def to_world(self, row: float, col: float) -> np.ndarray:
        forward = (self.spec.center - row) * self.spec.resolution
        left = (self.spec.center - col) * self.spec.resolution
        cos, sin = np.cos(self.pose[2]), np.sin(self.pose[2])
        return self.pose[:2] + np.array([forward * cos - left * sin, forward * sin + left * cos])

    def layer_at(self, xy) -> np.ndarray:
        cells = np.rint(self.to_cells(xy)).astype(int)
        inside = np.all((cells >= 0) & (cells < self.spec.size), axis=1)
        rows = np.clip(cells[:, 0], 0, self.spec.size - 1)
        cols = np.clip(cells[:, 1], 0, self.spec.size - 1)
        return np.where(inside, self.layers[rows, cols], UNKNOWN)

    def symbols(self, robot: bool = True) -> np.ndarray:
        grid = np.vectorize(LAYER_SYMBOLS.get)(self.layers).astype("<U1")
        overlay = self.route & np.isin(self.layers, (FREE, NEAR, UNKNOWN))
        grid[overlay] = ROUTE_SYMBOL
        if self.goal is not None:
            grid[self.goal] = GOAL_SYMBOL
        if robot:
            grid[self.spec.center, self.spec.center] = ROBOT_SYMBOL
        return grid

    def text(self) -> str:
        return "\n".join("".join(row) for row in self.symbols())


def build_costmap(spec: CostmapSpec, pose, scan: Scan, route: Route | None, goal) -> LocalCostmap:
    pose = np.asarray(pose, dtype=float)[:3]
    shape = (spec.size, spec.size)
    blank = LocalCostmap(
        spec, pose, np.full(shape, UNKNOWN), np.zeros(shape, bool), None, np.empty((0, 2))
    )
    layers = _inflate(spec, _ray_cast(blank, scan))
    goal_cell = _goal_cell(blank, goal, layers)
    route_mask, route_points = _route_overlay(blank, route)
    goal_point = None if goal is None else np.asarray(goal, dtype=float).ravel()[:2]
    return LocalCostmap(spec, pose, layers, route_mask, goal_cell, route_points, goal_point)


def _ray_cast(blank: LocalCostmap, scan: Scan) -> np.ndarray:
    spec = blank.spec
    layers = np.full((spec.size, spec.size), UNKNOWN)
    reach = np.minimum(scan.ranges, scan.range_max)
    samples = int(np.ceil(scan.range_max / spec.resolution * RAY_SAMPLES_PER_CELL))
    fractions = np.linspace(0.0, 1.0, samples, endpoint=False)
    bearing = scan.angles + scan.origin[2]
    distances = reach[:, None] * fractions[None, :]
    xs = scan.origin[0] + distances * np.cos(bearing)[:, None]
    ys = scan.origin[1] + distances * np.sin(bearing)[:, None]
    _mark(layers, blank, np.column_stack([xs.ravel(), ys.ravel()]), FREE)
    _mark(layers, blank, scan.points[scan.hits], OBSTACLE)
    return layers


def _mark(layers, blank: LocalCostmap, xy, value) -> None:
    if len(xy) == 0:
        return
    cells = np.rint(blank.to_cells(xy)).astype(int)
    inside = np.all((cells >= 0) & (cells < blank.spec.size), axis=1)
    layers[cells[inside, 0], cells[inside, 1]] = value


def _inflate(spec: CostmapSpec, layers: np.ndarray) -> np.ndarray:
    obstacle = layers == OBSTACLE
    if not obstacle.any():
        return layers
    distance = distance_transform_edt(~obstacle) * spec.resolution
    inflated = layers.copy()
    observed = np.isin(layers, (FREE, NEAR))
    inflated[observed & (distance <= spec.inflation + 1e-9)] = NEAR
    inflated[observed & (distance <= spec.robot_radius + 1e-9)] = TOO_CLOSE
    return inflated


def _goal_cell(blank: LocalCostmap, goal, layers) -> tuple[int, int] | None:
    if goal is None:
        return None
    row, col = np.rint(blank.to_cells(goal)[0]).astype(int)
    size = blank.spec.size
    if not (0 <= row < size and 0 <= col < size):
        return None
    if layers[row, col] in (OBSTACLE, TOO_CLOSE):
        return None
    return int(row), int(col)


def _route_overlay(blank: LocalCostmap, route: Route | None):
    spec = blank.spec
    mask = np.zeros((spec.size, spec.size), dtype=bool)
    if route is None:
        return mask, np.empty((0, 2))
    start, _ = route.project(blank.pose[:2])
    stations = np.append(np.arange(start, route.length, spec.resolution / 2), route.length)
    points = np.array([route.point_at(station) for station in stations])
    cells = np.rint(blank.to_cells(points)).astype(int)
    inside = np.all((cells >= 0) & (cells < spec.size), axis=1)
    visible = len(points) if inside.all() else int(np.argmin(inside))
    mask[cells[:visible, 0], cells[:visible, 1]] = True
    return mask, points[:visible]
