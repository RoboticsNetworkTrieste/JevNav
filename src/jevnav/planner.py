import heapq
import math
from dataclasses import dataclass

import numpy as np

from .occupancy import OccupancyGrid
from .route import Route

NEIGHBORS = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
)
SHORTCUT_PATIENCE = 12
GOAL_SNAP_RADIUS = 1.0


@dataclass(frozen=True)
class PlannerConfig:
    robot_radius: float
    safety_margin: float = 0.1
    comfort_distance: float = 0.6
    proximity_weight: float = 4.0


class RoutePlanner:
    def __init__(self, grid: OccupancyGrid, config: PlannerConfig):
        self.grid = grid
        self.config = config

    def plan(self, start, goal, extra_obstacles=None) -> Route | None:
        grid = self.grid if extra_obstacles is None else self.grid.with_points(extra_obstacles)
        gaps = grid.clearance - self.config.robot_radius - grid.resolution * 0.5
        start_cell = self._clamped(grid, grid.index_of(start))
        goal_cell = self._snap_goal(grid, gaps, goal)
        if goal_cell is None:
            return None
        cells = self._search(grid, gaps, start_cell, goal_cell)
        if cells is None:
            return None
        points = [grid.center_of(i, j) for i, j in cells]
        points[0] = np.asarray(start, dtype=float)
        points[-1] = (
            np.asarray(goal, dtype=float)
            if self._is_passable(gaps, grid.index_of(goal), grid)
            else points[-1]
        )
        cell_gaps = [gaps[i, j] for i, j in cells]
        return Route.through(self._shortcut(grid, gaps, np.array(points), cell_gaps))

    def _clamped(self, grid, cell):
        return (
            int(np.clip(cell[0], 0, grid.shape[0] - 1)),
            int(np.clip(cell[1], 0, grid.shape[1] - 1)),
        )

    def _is_passable(self, gaps, cell, grid) -> bool:
        return grid.inside(*cell) and gaps[cell] > 0

    def _snap_goal(self, grid, gaps, goal):
        cell = self._clamped(grid, grid.index_of(goal))
        if gaps[cell] > 0:
            return cell
        reach = math.ceil(GOAL_SNAP_RADIUS / grid.resolution)
        i0, j0 = cell
        best = None
        for i in range(max(0, i0 - reach), min(grid.shape[0], i0 + reach + 1)):
            for j in range(max(0, j0 - reach), min(grid.shape[1], j0 + reach + 1)):
                distance = math.hypot(i - i0, j - j0)
                if gaps[i, j] > 0 and distance <= reach and (best is None or distance < best[0]):
                    best = (distance, (i, j))
        return None if best is None else best[1]

    def _step_cost(self, gap: float) -> float:
        comfort = self.config.comfort_distance
        shortfall = min(max((comfort - gap) / comfort, 0.0), 1.0)
        return 1.0 + self.config.proximity_weight * shortfall * shortfall

    def _search(self, grid, gaps, start, goal):
        nx, ny = grid.shape
        best_cost = np.full((nx, ny), np.inf)
        parent = {}
        best_cost[start] = 0.0
        frontier = [(self._heuristic(start, goal), 0.0, start)]
        while frontier:
            _, cost, cell = heapq.heappop(frontier)
            if cell == goal:
                return self._backtrack(parent, cell)
            if cost > best_cost[cell]:
                continue
            for di, dj, length in NEIGHBORS:
                neighbor = (cell[0] + di, cell[1] + dj)
                if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny):
                    continue
                if gaps[neighbor] <= 0 and neighbor != goal:
                    continue
                candidate = cost + length * self._step_cost(gaps[neighbor])
                if candidate < best_cost[neighbor]:
                    best_cost[neighbor] = candidate
                    parent[neighbor] = cell
                    heapq.heappush(
                        frontier, (candidate + self._heuristic(neighbor, goal), candidate, neighbor)
                    )
        return None

    def _heuristic(self, cell, goal) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    def _backtrack(self, parent, cell):
        cells = [cell]
        while cell in parent:
            cell = parent[cell]
            cells.append(cell)
        return cells[::-1]

    def _shortcut(self, grid, gaps, points, cell_gaps) -> np.ndarray:
        kept = [points[0]]
        anchor = 0
        last = len(points) - 1
        while anchor < last:
            reachable = anchor + 1
            floor = cell_gaps[anchor + 1]
            misses = 0
            for candidate in range(anchor + 2, last + 1):
                floor = min(floor, cell_gaps[candidate])
                required = max(
                    min(self.config.comfort_distance, floor) - grid.resolution,
                    self.config.safety_margin,
                )
                if self._segment_gap(grid, points[anchor], points[candidate]) >= required:
                    reachable = candidate
                    misses = 0
                else:
                    misses += 1
                    if misses >= SHORTCUT_PATIENCE:
                        break
            kept.append(points[reachable])
            anchor = reachable
        return np.array(kept)

    def _segment_gap(self, grid, a, b) -> float:
        samples = max(2, math.ceil(np.linalg.norm(b - a) / (grid.resolution * 0.5)) + 1)
        line = a + np.linspace(0.0, 1.0, samples)[:, None] * (b - a)
        return (
            float(np.min(grid.clearance_at(line)))
            - self.config.robot_radius
            - grid.resolution * 0.5
        )
