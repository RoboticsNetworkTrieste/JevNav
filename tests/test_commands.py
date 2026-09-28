import numpy as np
import pytest

from jevnav.commands import by_id
from jevnav.kinematics import predict

from .conftest import SPEED, TURN_RATE


def test_twenty_four_distinct_commands_include_the_four_full_ones(commands):
    ids = [command.id for command in commands]
    assert len(ids) == 24
    assert len(set(ids)) == 24
    assert {"F", "B", "L", "R"} <= set(ids)


def test_full_commands_drive_straight_or_rotate_in_place(commands):
    table = by_id(commands)
    assert table["F"].action == [SPEED, 0.0]
    assert table["B"].action == [-SPEED, 0.0]
    assert table["L"].linear == 0.0
    assert table["L"].angular == pytest.approx(TURN_RATE)
    assert table["R"].angular == pytest.approx(-TURN_RATE)


@pytest.mark.parametrize(
    ("command_id", "forward_sign", "left_sign"),
    [("FL30", 1, 1), ("FR30", 1, -1), ("BL30", -1, 1), ("BR30", -1, -1)],
)
def test_partial_turns_travel_toward_their_named_side(
    commands, command_id, forward_sign, left_sign
):
    command = by_id(commands)[command_id]
    end = predict([0.0, 0.0, 0.0], command.linear, command.angular, 1.7, 0.1)
    assert np.sign(end[0]) == forward_sign
    assert np.sign(end[1]) == left_sign


def test_turn_rate_is_linear_in_the_direction_and_capped(commands):
    table = by_id(commands)
    assert table["FL15"].angular == pytest.approx(TURN_RATE / 6)
    assert table["FL75"].angular == pytest.approx(TURN_RATE * 75 / 90)
    assert table["BL15"].angular == pytest.approx(-TURN_RATE / 6)
    assert table["BR15"].angular == pytest.approx(TURN_RATE / 6)
    assert max(abs(command.angular) for command in commands) == pytest.approx(TURN_RATE)


def test_option_texts_are_fixed(commands):
    table = by_id(commands)
    assert table["FL30"].text() == "Forward, curving left 15 degrees per second."
    assert table["BR45"].text() == "Backward toward the rear-right, 45 degrees off straight back."
    assert table["L"].text() == "Full left: rotate in place to the left."
    assert table["B"].text() == "Full backward: straight back."
