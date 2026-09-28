import numpy as np
import pytest

from jevnav.commands import by_id
from jevnav.costmap import Scan
from jevnav.safety import SAFETY_STOP, ObstacleGeometry, SafetyLimits, first_safe, screen

from .conftest import SPEED, TURN_RATE, wall_scan

LIMITS = SafetyLimits(0.2, (-SPEED, SPEED), (-TURN_RATE, TURN_RATE))
POSE = [0.0, 0.0, 0.0]


def test_driving_into_a_close_wall_is_a_collision(commands):
    verdicts = screen(commands, POSE, 1.7, 0.1, wall_scan(0.5), LIMITS)
    assert verdicts["F"].reason == "collision"
    assert verdicts["B"].safe
    assert verdicts["L"].safe


def test_a_pass_closer_than_the_margin_is_unsafe(commands):
    scan = wall_scan(0.8)
    assert screen(commands, POSE, 1.7, 0.1, scan, LIMITS)["F"].safe
    strict = SafetyLimits(0.2, LIMITS.linear, LIMITS.angular, margin=0.2)
    verdict = screen(commands, POSE, 1.7, 0.1, scan, strict)["F"]
    assert verdict.reason == "clearance"
    assert verdict.clearance == pytest.approx(0.175, abs=0.01)


def test_end_clearance_is_the_gap_where_the_command_ends(commands):
    verdicts = screen(commands, POSE, 1.7, 0.1, wall_scan(0.8), LIMITS)
    assert verdicts["B"].clearance == pytest.approx(0.6, abs=0.01)
    assert verdicts["B"].end_clearance == pytest.approx(1.025, abs=0.01)
    assert verdicts["F"].end_clearance == pytest.approx(verdicts["F"].clearance)
    assert verdicts["L"].end_clearance == pytest.approx(verdicts["L"].clearance)


def test_commands_outside_the_velocity_limits_are_infeasible(commands):
    slow = SafetyLimits(0.2, (-0.1, 0.1), LIMITS.angular)
    verdicts = screen(commands, POSE, 1.7, 0.1, wall_scan(9.0), slow)
    assert verdicts["F"].reason == "kinematics"
    assert verdicts["L"].safe


def test_the_disc_cannot_slip_between_two_close_hits():
    spread = np.arctan2(0.15, 1.0)
    scan = Scan(
        origin=np.zeros(3),
        ranges=np.array([np.hypot(1.0, 0.15), np.hypot(1.0, 0.15), 5.0]),
        angles=np.array([-spread, spread, np.pi / 2]),
        range_max=5.0,
    )
    geometry = ObstacleGeometry.from_scan(scan, 0.4)
    assert geometry.distance(np.array([[1.0, 0.0]]))[0] == pytest.approx(0.0, abs=1e-9)


def test_final_validation_falls_back_to_the_next_safe_ranked_command(commands):
    table = by_id(commands)
    options = [table["F"], table["B"], table["L"]]
    scan = wall_scan(0.5)
    command, position = first_safe(["F", "B", "L"], options, POSE, 1.7, 0.1, scan, LIMITS)
    assert (command.id, position) == ("B", 1)
    stop, skipped = first_safe(["F"], options, POSE, 1.7, 0.1, scan, LIMITS)
    assert (stop, skipped) == (SAFETY_STOP, 1)
    assert first_safe(["FL30", "L"], options, POSE, 1.7, 0.1, scan, LIMITS)[0].id == "L"
