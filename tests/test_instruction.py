import numpy as np
import pytest

from jevnav.deciders import HeuristicDecider
from jevnav.navigator import Navigator, NavigatorConfig, Observer
from jevnav.prompt import EvidenceFormat, compose
from jevnav.route import Route
from jevnav.safety import SafetyLimits, screen
from jevnav.scenario import open_env, resolve_scenario
from jevnav.visualizer import PRESETS, Visualizer

from .conftest import SPEED, TURN_RATE, wall_scan
from .test_navigator import ScriptedDecider

KEEP_CLEAR = PRESETS["Keep clear"]


def request(commands, wall_costmap, evidence_format, instruction):
    route = Route.through([[0.0, 0.0], [0.0, 3.0]])
    limits = SafetyLimits(0.2, (-SPEED, SPEED), (-TURN_RATE, TURN_RATE))
    verdicts = screen(commands, [0.0, 0.0, 0.0], 1.7, 0.1, wall_scan(1.0), limits)
    return compose(
        0,
        0,
        wall_costmap,
        commands,
        1.7,
        np.random.default_rng(0),
        evidence_format,
        candidates=[command for command in commands if verdicts[command.id].safe],
        route=route,
        verdicts=verdicts,
        instruction=instruction,
    )


@pytest.mark.parametrize("evidence_format", list(EvidenceFormat))
def test_the_instruction_joins_the_question_not_the_evidence(
    commands, wall_costmap, evidence_format
):
    plain = request(commands, wall_costmap, evidence_format, None)
    steered = request(commands, wall_costmap, evidence_format, KEEP_CLEAR)
    assert steered.evidence == plain.evidence
    assert steered.criteria() == plain.criteria()
    assert steered.instructions.startswith(plain.instructions)
    assert steered.instructions.endswith(f"preferences given: {KEEP_CLEAR}")
    assert steered.instruction == KEEP_CLEAR and plain.instruction is None


class ChangeOfMind(Observer):
    def on_decision(self, navigator, record, request):
        if record.index == 1:
            navigator.instruction = "Hurry."


def test_an_instruction_changed_mid_run_reaches_the_next_request(env_factory):
    config = NavigatorConfig(hold=0.5, base_time_limit=3.0, time_limit_factor=0.0)
    report = Navigator(
        env_factory("open"), ScriptedDecider(latency=0.3), config, observer=ChangeOfMind()
    ).run()
    steered = [record.index for record in report.decisions if record.instruction == "Hurry."]
    assert steered and min(steered) == 2
    assert all(record.instruction is None for record in report.decisions if record.index < 2)


def test_the_viewer_box_and_buttons_steer_the_navigator():
    env = open_env(resolve_scenario("open"), figure=True)
    try:
        viewer = Visualizer(env, interactive=True, speed=0)
        config = NavigatorConfig(hold=0.5, base_time_limit=1.0, time_limit_factor=0.0)
        navigator = Navigator(env, HeuristicDecider(), config, observer=viewer)
        navigator.run()
        viewer.textbox.set_val("Keep to the middle.")
        assert navigator.instruction == "Keep to the middle."
        viewer.textbox.set_val(PRESETS["Clear"])
        assert navigator.instruction is None
        assert "instruction: none" in viewer.status.get_text()
    finally:
        env.end(0)
