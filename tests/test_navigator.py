import time

import numpy as np
import pytest

from jevnav.deciders import Decision, HeuristicDecider
from jevnav.kinematics import predict
from jevnav.navigator import Navigator, NavigatorConfig, Observer, Phase
from jevnav.prompt import EvidenceFormat, compose


class ScriptedDecider:
    name = "scripted"

    def __init__(self, latency: float, command_id: str = "F"):
        self.latency = latency
        self.command_id = command_id

    def decide(self, request):
        probabilities = {command.id: 0.0 for command in request.options}
        probabilities[self.command_id] = 1.0
        return Decision(self.command_id, probabilities, self.latency)


def short_run(env, decider, hold, seconds=3.0):
    config = NavigatorConfig(hold=hold, base_time_limit=seconds, time_limit_factor=0.0)
    return Navigator(env, decider, config, scenario="open", seed=0).run()


def test_prediction_matches_ir_sim_step_for_step(env_factory, commands):
    env = env_factory("open")
    worst = 0.0
    for command in commands[::5]:
        pose = env.robot.state[:3, 0].copy()
        expected = predict(pose, command.linear, command.angular, 1.7, 0.1)
        for _ in range(17):
            env.step(command.action)
        worst = max(worst, float(np.max(np.abs(expected - env.robot.state[:3, 0]))))
    assert worst < 1e-9


def test_robot_stands_still_until_the_first_decision_arrives(env_factory):
    report = short_run(env_factory("open"), ScriptedDecider(latency=0.35), hold=0.5)
    first = report.decisions[0]
    assert first.applied_at == pytest.approx(0.4)
    assert first.overrun == 0.0
    assert report.trajectory[0] == report.trajectory[4]
    assert report.trajectory[4] != report.trajectory[5]


def test_decisions_switch_exactly_at_the_end_of_each_hold(env_factory):
    report = short_run(env_factory("open"), ScriptedDecider(latency=0.3), hold=0.5)
    applied = [record.applied_at for record in report.applied]
    assert applied[:4] == pytest.approx([0.3, 0.8, 1.3, 1.8])
    assert [record.sent_at for record in report.applied][:4] == pytest.approx([0.0, 0.3, 0.8, 1.3])
    assert report.overruns == 0


def test_a_late_decision_is_an_overrun_that_keeps_the_previous_command(env_factory):
    report = short_run(env_factory("open"), ScriptedDecider(latency=0.45), hold=0.3)
    second = report.applied[1]
    assert report.applied[0].applied_at == pytest.approx(0.5)
    assert second.sent_at == pytest.approx(0.5)
    assert second.applied_at == pytest.approx(1.0)
    assert second.overrun == pytest.approx(0.2)
    assert report.overruns >= 1


def test_each_costmap_is_centred_where_the_robot_is_when_its_command_starts(env_factory):
    report = short_run(
        env_factory("open"), ScriptedDecider(latency=0.3, command_id="FL45"), hold=0.5
    )
    for record in report.applied[1:4]:
        step = round(record.applied_at / 0.1)
        assert record.predicted_pose[:2] == pytest.approx(report.trajectory[step], abs=1e-3)


def test_warm_up_sets_the_hold_from_the_measured_latency(env_factory):
    config = NavigatorConfig(hold_factor=1.2, base_time_limit=1.0, time_limit_factor=0.0)
    navigator = Navigator(
        env_factory("open"), ScriptedDecider(latency=1.0), config, scenario="open"
    )
    report = navigator.run()
    assert report.warmup_latencies == [1.0, 1.0, 1.0]
    assert report.hold == pytest.approx(1.2)


def test_request_lists_all_commands_in_a_seeded_shuffle(wall_costmap, commands):
    first = compose(0, 0, wall_costmap, commands, 1.7, np.random.default_rng(7))
    again = compose(0, 0, wall_costmap, commands, 1.7, np.random.default_rng(7))
    assert [command.id for command in first.options] == [command.id for command in again.options]
    assert sorted(command.id for command in first.options) == sorted(
        command.id for command in commands
    )
    assert [command.id for command in first.options] != [command.id for command in commands]
    assert first.evidence.endswith(wall_costmap.text())


@pytest.mark.parametrize("scenario", ["open", "slalom"])
def test_heuristic_reaches_the_goal_without_collision(env_factory, scenario):
    report = Navigator(
        env_factory(scenario), HeuristicDecider(), NavigatorConfig(hold=1.7), scenario=scenario
    ).run()
    assert report.outcome == "arrived"
    assert report.efficiency > 0.8


class SlowingDecider(ScriptedDecider):
    def __init__(self, latencies):
        super().__init__(latency=latencies[-1])
        self.latencies = list(latencies)

    def decide(self, request):
        self.latency = self.latencies.pop(0) if len(self.latencies) > 1 else self.latencies[0]
        return super().decide(request)


def test_hold_stretches_when_the_model_slows_down(env_factory):
    decider = SlowingDecider([0.5, 0.5, 0.5, 0.5, 0.5, 1.0])
    config = NavigatorConfig(hold_factor=1.2, base_time_limit=6.0, time_limit_factor=0.0)
    report = Navigator(env_factory("open"), decider, config, scenario="open").run()
    applied = report.applied
    assert report.hold == pytest.approx(0.6)
    assert [record.hold for record in applied[:4]] == pytest.approx([0.6, 0.6, 1.2, 1.2])
    assert [record.applied_at for record in applied[:4]] == pytest.approx([0.5, 1.1, 2.1, 3.3])
    assert applied[2].overrun == pytest.approx(0.4)
    assert report.overruns == 1


def test_the_first_costmap_already_sees_the_obstacles(env_factory):
    report = short_run(env_factory("open"), ScriptedDecider(latency=0.3), hold=0.5, seconds=0.5)
    assert "#" in report.decisions[0].evidence


def test_heuristic_reaches_the_goal_through_the_simulation_pipeline(env_factory):
    config = NavigatorConfig(hold=1.7, evidence_format=EvidenceFormat.SIMULATION)
    report = Navigator(env_factory("slalom"), HeuristicDecider(), config, scenario="slalom").run()
    assert report.outcome == "arrived"
    assert report.safety_stops == 0
    assert all(record.removed is not None for record in report.decisions)


class ThinkingDecider(ScriptedDecider):
    def __init__(self, seconds: float):
        super().__init__(latency=seconds)

    def decide(self, request):
        time.sleep(self.latency)
        return super().decide(request)


class PauseWatcher(Observer):
    def __init__(self):
        self.waits = []

    def on_wait(self, navigator):
        self.waits.append((navigator.time, navigator.phase, navigator.waited))


def fake_timing_run(env, decider, observer=None, seconds=5.0):
    config = NavigatorConfig(sim_latency=1.0, base_time_limit=seconds, time_limit_factor=0.0)
    return Navigator(env, decider, config, scenario="open", seed=0, observer=observer).run()


def test_fake_timing_counts_every_answer_as_the_simulated_latency(env_factory):
    report = fake_timing_run(env_factory("open"), ScriptedDecider(latency=7.5))
    applied = report.applied
    assert report.hold == pytest.approx(1.2)
    assert report.warmup_latencies == []
    assert [record.applied_at for record in applied[:4]] == pytest.approx([1.0, 2.2, 3.4, 4.6])
    assert [record.hold for record in applied[:4]] == pytest.approx([1.2] * 4)
    assert report.overruns == 0
    assert all(record.latency == 7.5 for record in report.decisions)
    assert report.summary()["sim_latency_s"] == 1.0


def test_the_simulation_pauses_while_a_slow_model_thinks(env_factory):
    watcher = PauseWatcher()
    report = fake_timing_run(env_factory("open"), ThinkingDecider(0.3), watcher, seconds=3.0)
    assert [record.applied_at for record in report.applied[:2]] == pytest.approx([1.0, 2.2])
    assert watcher.waits
    assert all(phase == Phase.WAITING for _, phase, _ in watcher.waits)
    frozen = {}
    for sim_time, _, waited in watcher.waits:
        frozen.setdefault(round(sim_time, 6), []).append(waited)
    assert all(np.all(np.diff(waited) > 0) for waited in frozen.values())
    assert set(frozen) <= {0.0, 1.0, 2.2}
