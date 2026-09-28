from dataclasses import dataclass

import numpy as np

from .commands import Command
from .costmap import Scan
from .kinematics import rollout

SAFETY_STOP = Command("STOP", 0, 0.0, 0.0)


@dataclass(frozen=True)
class SafetyLimits:
    radius: float
    linear: tuple[float, float]
    angular: tuple[float, float]
    margin: float = 0.05

    @classmethod
    def from_robot(cls, robot, margin: float = 0.05) -> "SafetyLimits":
        low = np.asarray(robot.vel_min, dtype=float).ravel()
        high = np.asarray(robot.vel_max, dtype=float).ravel()
        return cls(float(robot.radius), (low[0], high[0]), (low[1], high[1]), margin)


@dataclass(frozen=True)
class Verdict:
    safe: bool
    reason: str | None
    clearance: float
    end_clearance: float


@dataclass(frozen=True)
class ObstacleGeometry:
    starts: np.ndarray
    ends: np.ndarray

    @classmethod
    def from_scan(cls, scan: Scan, join_below: float) -> "ObstacleGeometry":
        points = scan.points
        hits = scan.hits
        indices = np.flatnonzero(hits)
        following = (indices + 1) % len(hits)
        joined = hits[following] & (
            np.linalg.norm(points[following] - points[indices], axis=1) < join_below
        )
        starts = np.vstack([points[indices], points[indices[joined]]])
        ends = np.vstack([points[indices], points[following[joined]]])
        return cls(starts, ends)

    def distance(self, xy: np.ndarray) -> np.ndarray:
        if len(self.starts) == 0:
            return np.full(len(xy), np.inf)
        direction = self.ends - self.starts
        squared = np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-12)
        offset = xy[:, None, :] - self.starts[None, :, :]
        along = np.clip(np.einsum("sij,ij->si", offset, direction) / squared, 0.0, 1.0)
        closest = self.starts[None, :, :] + along[..., None] * direction[None, :, :]
        return np.min(np.linalg.norm(xy[:, None, :] - closest, axis=2), axis=1)


def check(
    command: Command,
    pose,
    hold: float,
    step: float,
    obstacles: ObstacleGeometry,
    limits: SafetyLimits,
) -> Verdict:
    if not (
        limits.linear[0] - 1e-9 <= command.linear <= limits.linear[1] + 1e-9
        and limits.angular[0] - 1e-9 <= command.angular <= limits.angular[1] + 1e-9
    ):
        return Verdict(False, "kinematics", float("nan"), float("nan"))
    path = rollout(pose, command.linear, command.angular, hold, step)
    gaps = obstacles.distance(path[:, :2]) - limits.radius
    clearance, end_clearance = float(np.min(gaps)), float(gaps[-1])
    if clearance < 0:
        return Verdict(False, "collision", clearance, end_clearance)
    if clearance < limits.margin:
        return Verdict(False, "clearance", clearance, end_clearance)
    return Verdict(True, None, clearance, end_clearance)


def screen(commands, pose, hold, step, scan: Scan, limits: SafetyLimits) -> dict[str, Verdict]:
    obstacles = ObstacleGeometry.from_scan(scan, 2 * limits.radius)
    return {command.id: check(command, pose, hold, step, obstacles, limits) for command in commands}


def first_safe(
    ranking, options, pose, hold: float, step: float, scan: Scan, limits: SafetyLimits
) -> tuple[Command, int]:
    obstacles = ObstacleGeometry.from_scan(scan, 2 * limits.radius)
    offered = {command.id: command for command in options}
    ranked = [command_id for command_id in ranking if command_id in offered]
    for position, command_id in enumerate(ranked):
        if check(offered[command_id], pose, hold, step, obstacles, limits).safe:
            return offered[command_id], position
    return SAFETY_STOP, len(ranked)
