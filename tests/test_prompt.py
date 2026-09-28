import re
from dataclasses import replace

import numpy as np
import pytest

from jevnav.costmap import CostmapSpec, build_costmap
from jevnav.prompt import EvidenceFormat, compose, describe
from jevnav.route import Route

from .conftest import wall_scan

POINT = re.compile(r"\((-?\d+\.\d), (-?\d+\.\d)\)")


def line(text: str, prefix: str) -> str:
    return next(row for row in text.split("\n") if row.startswith(prefix))


def points(row: str) -> np.ndarray:
    return np.array([[float(x), float(y)] for x, y in POINT.findall(row.split(": ", 1)[1])])


@pytest.fixture
def facing_left():
    wall_on_the_right = replace(wall_scan(1.0), origin=np.array([1.0, 2.0, 0.0]))
    route = Route.through([[0.0, 0.0], [0.0, 5.0]])
    return build_costmap(CostmapSpec(), [1.0, 2.0, np.pi / 2], wall_on_the_right, route, [0.0, 4.5])


def test_text_gives_the_world_pose_and_the_goal_relative_to_the_robot(facing_left):
    text = describe(facing_left)
    assert line(text, "Robot:") == "Robot: at (1.0, 2.0) in the world, heading 90 degrees."
    assert line(text, "Goal:") == "Goal: (2.5, 1.0), 2.7 m away."


def test_obstacles_are_the_wall_cells_in_the_robot_frame_nearest_first(wall_costmap):
    obstacles = points(line(describe(wall_costmap), "Obstacles"))
    assert len(obstacles) == np.count_nonzero(wall_costmap.symbols() == "#")
    assert np.all(obstacles[:, 0] == 1.0)
    assert obstacles[0].tolist() == [1.0, 0.0]
    assert np.all(np.diff(np.hypot(obstacles[:, 0], obstacles[:, 1])) >= 0)


def test_path_is_resampled_every_0_4_m_up_to_the_window_edge(wall_costmap):
    path = points(line(describe(wall_costmap), "Planned path"))
    assert path[:, 0].tolist() == [0.0] * 6
    assert path[:, 1].tolist() == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])


def test_rotating_the_robot_rotates_every_coordinate(facing_left):
    text = describe(facing_left)
    path = points(line(text, "Planned path"))
    assert path[:, 0].tolist() == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    assert np.all(path[:, 1] == 1.0)
    assert np.all(points(line(text, "Obstacles"))[:, 1] == -1.0)


def test_nothing_in_view_is_said_in_words():
    empty = build_costmap(CostmapSpec(), [0.0, 0.0, 0.0], wall_scan(9.0), None, None)
    text = describe(empty)
    assert line(text, "Goal:") == "Goal: not given."
    assert line(text, "Planned path").endswith(": none in view")
    assert line(text, "Obstacles").endswith(": none")


def test_text_request_has_no_grid_and_word_instructions(wall_costmap, commands):
    rng = np.random.default_rng(0)
    request = compose(0, 0, wall_costmap, commands, 1.7, rng, EvidenceFormat.TEXT)
    assert wall_costmap.text() not in request.evidence
    assert "closer than 0.2 m to the robot centre is a collision" in request.instructions
    assert "keep at least 0.6 m" in request.instructions
    assert "(*)" not in request.instructions
