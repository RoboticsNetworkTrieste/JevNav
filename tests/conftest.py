import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from jevnav.commands import command_set
from jevnav.costmap import CostmapSpec, Scan, build_costmap
from jevnav.route import Route
from jevnav.scenario import open_env, resolve_scenario

SPEED = 0.25
TURN_RATE = math.radians(45)


@pytest.fixture
def commands():
    return command_set(SPEED, TURN_RATE)


def wall_scan(
    distance: float, half_width: float = 1.0, beams: int = 360, range_max: float = 5.0
) -> Scan:
    angles = np.linspace(-np.pi, np.pi, beams)
    ranges = np.full(beams, range_max)
    ahead = np.cos(angles) > 1e-9
    along = np.abs(np.tan(angles) * distance) <= half_width
    ranges[ahead & along] = distance / np.cos(angles[ahead & along])
    return Scan(
        origin=np.zeros(3), ranges=np.minimum(ranges, range_max), angles=angles, range_max=range_max
    )


@pytest.fixture
def wall_costmap():
    route = Route.through([[0.0, 0.0], [0.0, 3.0]])
    return build_costmap(CostmapSpec(), [0.0, 0.0, 0.0], wall_scan(1.0), route, [0.0, 1.6])


@pytest.fixture
def env_factory():
    envs = []

    def make(name: str, seed: int = 0):
        env = open_env(resolve_scenario(name), seed=seed)
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.end(0)
