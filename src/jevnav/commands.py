import math
from dataclasses import dataclass

DIRECTION_STEP_DEGREES = 15


@dataclass(frozen=True)
class Command:
    id: str
    direction: float
    linear: float
    angular: float

    @property
    def action(self) -> list[float]:
        return [self.linear, self.angular]

    @property
    def turn_rate_degrees(self) -> float:
        return math.degrees(self.angular)

    def text(self) -> str:
        rate = f"{abs(self.turn_rate_degrees):g}"
        side = "left" if self.direction > 0 else "right"
        offset = 180 - abs(self.direction)
        if self.id == "F":
            return "Full forward: straight ahead."
        if self.id == "B":
            return "Full backward: straight back."
        if self.id in ("L", "R"):
            return f"Full {side}: rotate in place to the {side}."
        if self.linear > 0:
            return f"Forward, curving {side} {rate} degrees per second."
        return f"Backward toward the rear-{side}, {offset:g} degrees off straight back."


def command_id(direction: int) -> str:
    if direction == 0:
        return "F"
    if abs(direction) == 180:
        return "B"
    if direction == 90:
        return "L"
    if direction == -90:
        return "R"
    if 0 < direction < 90:
        return f"FL{direction}"
    if -90 < direction < 0:
        return f"FR{-direction}"
    if direction > 90:
        return f"BL{180 - direction}"
    return f"BR{180 + direction}"


def command_for(direction: int, speed: float, turn_rate: float) -> Command:
    magnitude = abs(direction)
    sign = 1 if direction > 0 else -1
    if magnitude == 90:
        return Command(command_id(direction), direction, 0.0, sign * turn_rate)
    if magnitude < 90:
        return Command(command_id(direction), direction, speed, turn_rate * direction / 90)
    angular = 0.0 if magnitude == 180 else -sign * turn_rate * (180 - magnitude) / 90
    return Command(command_id(direction), 180 if magnitude == 180 else direction, -speed, angular)


def command_set(speed: float, turn_rate: float) -> tuple[Command, ...]:
    directions = range(-165, 181, DIRECTION_STEP_DEGREES)
    ordered = sorted(directions, key=lambda value: value % 360)
    return tuple(command_for(direction, speed, turn_rate) for direction in ordered)


def by_id(commands) -> dict[str, Command]:
    return {command.id: command for command in commands}
