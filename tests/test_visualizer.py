import matplotlib
import matplotlib.pyplot as plt
import pytest

from jevnav.deciders import HeuristicDecider
from jevnav.navigator import Navigator, NavigatorConfig
from jevnav.scenario import open_env, resolve_scenario
from jevnav.visualizer import Visualizer

SHORT_RUN = NavigatorConfig(hold=0.5, base_time_limit=1.0, time_limit_factor=0.0)


@pytest.fixture
def figure_env():
    env = open_env(resolve_scenario("open"), figure=True)
    with matplotlib.rc_context({"figure.raise_window": True}):
        yield env
    env.end(0)


def raise_flags_at_each_render(env) -> list[bool]:
    flags = []
    render = env.render

    def recording_render(*args, **kwargs):
        flags.append(plt.rcParams["figure.raise_window"])
        render(*args, **kwargs)

    env.render = recording_render
    return flags


def test_the_window_comes_to_the_front_when_the_run_starts_and_never_again(figure_env):
    flags = raise_flags_at_each_render(figure_env)
    viewer = Visualizer(figure_env, interactive=True, speed=0)
    Navigator(figure_env, HeuristicDecider(), SHORT_RUN, observer=viewer).run()
    assert len(flags) > 2
    assert flags[0] is True
    assert not any(flags[1:])


def test_recordings_leave_the_window_setting_alone(figure_env):
    flags = raise_flags_at_each_render(figure_env)
    recorder = Visualizer(figure_env, interactive=False)
    Navigator(figure_env, HeuristicDecider(), SHORT_RUN, observer=recorder).run()
    assert all(flags)


def test_hold_open_returns_once_the_window_is_closed(figure_env):
    viewer = Visualizer(figure_env, interactive=True, speed=0)
    Navigator(figure_env, HeuristicDecider(), SHORT_RUN, observer=viewer).run()
    plt.close(viewer.figure)
    viewer.hold_open()
