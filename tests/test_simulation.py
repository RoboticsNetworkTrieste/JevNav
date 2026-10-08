import re

import numpy as np
import pytest

from jevnav import navigator
from jevnav.commands import by_id
from jevnav.costmap import CostmapSpec, build_costmap
from jevnav.deciders import JevDecider
from jevnav.navigator import Navigator, NavigatorConfig
from jevnav.prompt import EvidenceFormat, compose
from jevnav.route import Route
from jevnav.safety import SafetyLimits, Verdict, screen
from jevnav.simulation import simulate_commands

from .conftest import SPEED, TURN_RATE, wall_scan

LIMITS = SafetyLimits(0.2, (-SPEED, SPEED), (-TURN_RATE, TURN_RATE))
STRAIGHT = Route.through([[0.0, 0.0], [5.0, 0.0]])


def simulation_request(
    commands, wall: float, seed: int = 0, evidence_format=EvidenceFormat.SIMULATION
):
    scan = wall_scan(wall)
    costmap = build_costmap(CostmapSpec(), [0.0, 0.0, 0.0], scan, STRAIGHT, [5.0, 0.0])
    verdicts = screen(commands, [0.0, 0.0, 0.0], 2.2, 0.1, scan, LIMITS)
    return compose(
        0,
        0,
        costmap,
        commands,
        2.2,
        np.random.default_rng(seed),
        evidence_format,
        candidates=[command for command in commands if verdicts[command.id].safe],
        route=STRAIGHT,
        verdicts=verdicts,
    )


def results(description: str) -> tuple[float, float, float]:
    found = re.search(
        r"After it: (\d+\.\d+) m to go along the route, (\d+\.\d+) m off the route, "
        r"(\d+\.\d+) m straight to the goal;",
        description,
    )
    return tuple(float(number) for number in found.groups())


def test_colliding_commands_are_not_offered(commands):
    request = simulation_request(commands, wall=0.6)
    offered = {command.id for command in request.options}
    assert "F" not in offered
    assert request.verdicts["F"].reason == "collision"
    assert {"B", "L", "R"} <= offered
    assert set(request.criteria()) == offered
    assert request.criteria()["B"].endswith("; closest obstacle 0.95 m.")
    assert request.criteria()["L"].endswith("; closest obstacle 0.40 m.")


def test_each_option_carries_its_simulated_distances(commands):
    request = simulation_request(commands, wall=9.0)
    criteria = request.criteria()
    assert len(criteria) == 24
    assert criteria["F"].startswith("Full forward: straight ahead. After it: 4.45 m to go")
    assert results(criteria["F"]) == pytest.approx((4.45, 0.0, 4.45))
    assert results(criteria["B"]) == pytest.approx((5.0, 0.55, 5.55))
    assert results(criteria["L"]) == results(criteria["R"]) == (5.0, 0.0, 5.0)
    assert results(criteria["FL45"])[1] > results(criteria["FL15"])[1] > 0.0
    assert not any("(" in text for text in criteria.values())
    assert criteria["F"].endswith("; no obstacle in lidar range.")
    assert "Now:" not in request.evidence
    assert "measured where the robot is when the command ends" in request.evidence
    assert request.instructions == "Which command should the robot hold for the next 2.2 s?"


def test_route_to_go_does_not_jump_to_a_nearby_later_leg(commands):
    hairpin = Route.through([[0.0, 0.0], [3.0, 0.0], [3.0, 0.5], [0.0, 0.5]])
    fl75 = by_id(commands)["FL75"]
    outlooks = simulate_commands([fl75], [0.0, 0.0, 0.0], 2.2, 0.1, hairpin, [0.0, 0.5])
    assert outlooks["FL75"].route_to_go > 5.0


def test_the_model_gets_one_choice_question_with_the_results(commands):
    request = simulation_request(commands, wall=0.6)
    body = JevDecider().request_body(request)
    question = body["questions"]["command"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == [command.id for command in request.options]
    assert all("After it:" in text for text in question["criteria"].values())
    assert body["state"] == request.evidence


class NeverCalled:
    name = "never"

    def decide(self, request):
        raise AssertionError("The model must not be asked when no command is safe")


def test_no_safe_command_means_a_safety_stop_without_asking(env_factory, monkeypatch):
    def nothing_safe(commands, *args):
        return {command.id: Verdict(False, "collision", -1.0, -1.0) for command in commands}

    monkeypatch.setattr(navigator, "screen", nothing_safe)
    config = NavigatorConfig(
        hold=0.5,
        base_time_limit=2.0,
        time_limit_factor=0.0,
        evidence_format=EvidenceFormat.SIMULATION,
    )
    report = Navigator(env_factory("open"), NeverCalled(), config, scenario="open").run()
    assert report.applied and all(record.executed == "STOP" for record in report.applied)
    assert report.distance_traveled == pytest.approx(0.0)
    assert report.safety_stops == len(report.applied)
