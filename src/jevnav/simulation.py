from dataclasses import dataclass

import numpy as np

from .kinematics import rollout
from .route import Route

REACH_MARGIN = 1.0


@dataclass(frozen=True)
class Outlook:
    route_to_go: float
    off_route: float
    goal_distance: float


def simulate_commands(
    commands, pose, hold: float, step: float, route: Route, goal
) -> dict[str, Outlook]:
    pose = np.asarray(pose, dtype=float)
    goal = np.asarray(goal, dtype=float)[:2]
    start, _ = route.project(pose[:2])
    reach = max((abs(command.linear) for command in commands), default=0.0) * hold + REACH_MARGIN
    results = {}
    for command in commands:
        end = rollout(pose, command.linear, command.angular, hold, step)[-1, :2]
        station, off_route = route.project(end, lower=start - reach, upper=start + reach)
        results[command.id] = Outlook(
            route.length - station, off_route, float(np.linalg.norm(goal - end))
        )
    return results
