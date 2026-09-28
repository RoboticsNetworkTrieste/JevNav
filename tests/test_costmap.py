import numpy as np

from jevnav.costmap import CostmapSpec, build_costmap
from jevnav.kinematics import predict
from jevnav.route import Route

from .conftest import wall_scan

CENTER = 10


def test_text_is_a_square_grid_with_the_robot_in_the_centre(wall_costmap):
    rows = wall_costmap.text().split("\n")
    assert len(rows) == 21
    assert all(len(row) == 21 for row in rows)
    assert rows[CENTER][CENTER] == "R"


def test_wall_ahead_is_inflated_and_occludes_what_is_behind(wall_costmap):
    grid = wall_costmap.symbols()
    assert grid[CENTER - 5, CENTER] == "#"
    assert grid[CENTER - 4, CENTER] == "x"
    assert grid[CENTER - 3, CENTER] == "+"
    assert grid[CENTER - 1, CENTER] == "."
    assert set(grid[: CENTER - 5, CENTER]) == {"?"}


def test_route_and_goal_are_drawn_to_the_left(wall_costmap):
    grid = wall_costmap.symbols()
    assert grid[CENTER, CENTER - 8] == "G"
    assert set(grid[CENTER, CENTER - 7 : CENTER]) == {"*"}
    assert wall_costmap.target[0] == 0.0
    assert abs(wall_costmap.target[1] - 2.0) <= 0.11


def test_rotating_the_pose_rotates_the_map():
    route = Route.through([[0.0, 0.0], [0.0, 3.0]])
    facing_left = build_costmap(CostmapSpec(), [0.0, 0.0, np.pi / 2], wall_scan(1.0), route, None)
    grid = facing_left.symbols()
    assert grid[CENTER, CENTER + 5] == "#"
    assert grid[CENTER - 3, CENTER] == "*"


def test_map_centred_on_a_predicted_pose_keeps_obstacles_in_world_place():
    scan = wall_scan(1.0)
    ahead = predict([0.0, 0.0, 0.0], 0.25, 0.0, 1.6, 0.1)
    costmap = build_costmap(CostmapSpec(), ahead, scan, None, None)
    assert costmap.symbols()[CENTER - 3, CENTER] == "#"
