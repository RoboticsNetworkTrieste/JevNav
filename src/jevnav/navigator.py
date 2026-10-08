import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from statistics import median

import numpy as np

from .commands import DIRECTION_STEP_DEGREES, Command, command_set
from .costmap import CostmapSpec, LocalCostmap, Scan, build_costmap
from .costmap_image import CELL_PIXELS, check_cell
from .deciders import SAFETY, Decider, Decision, safety_stop
from .kinematics import rollout
from .planner import PlannerConfig, RoutePlanner
from .prompt import SIMULATED, DecisionRequest, EvidenceFormat, compose
from .report import DecisionRecord, EpisodeReport
from .route import Route
from .safety import SAFETY_STOP, SafetyLimits, first_safe, screen
from .scenario import occupancy_from_env

WAIT_POLL = 0.05


class Outcome(StrEnum):
    ARRIVED = "arrived"
    COLLIDED = "collided"
    TIMED_OUT = "timed_out"
    UNREACHABLE = "unreachable"


class Phase(StrEnum):
    PLANNING = "planning"
    WARM_UP = "warm-up"
    AWAITING_FIRST = "awaiting first decision"
    EXECUTING = "executing"
    OVERRUN = "overrun"
    WAITING = "waiting for the model"
    FINISHED = "finished"


@dataclass(frozen=True)
class NavigatorConfig:
    hold: float | None = None
    hold_factor: float = 1.2
    warmup_requests: int = 3
    latency_window: int = 5
    costmap: CostmapSpec = field(default_factory=CostmapSpec)
    evidence_format: EvidenceFormat = EvidenceFormat.GRID
    clearance_margin: float = 0.05
    instruction: str | None = None
    image_cell: int = CELL_PIXELS
    sim_latency: float | None = None
    safety_margin: float = 0.1
    comfort_distance: float = 0.6
    replan_offset: float = 1.0
    stall_window: float = 6.0
    stall_progress: float = 0.3
    base_time_limit: float = 60.0
    time_limit_factor: float = 3.0
    grid_resolution: float | None = None


@dataclass
class Pending:
    request: DecisionRequest
    future: Future
    record: DecisionRecord | None = None


class Observer:
    def on_start(self, navigator: "Navigator") -> None:
        pass

    def on_request(self, navigator: "Navigator", request: DecisionRequest) -> None:
        pass

    def on_decision(
        self, navigator: "Navigator", record: DecisionRecord, request: DecisionRequest
    ) -> None:
        pass

    def on_step(self, navigator: "Navigator") -> None:
        pass

    def on_wait(self, navigator: "Navigator") -> None:
        pass

    def on_end(self, navigator: "Navigator") -> None:
        pass


class Navigator:
    def __init__(
        self,
        env,
        decider: Decider,
        config: NavigatorConfig | None = None,
        *,
        scenario: str = "scenario",
        seed: int | None = None,
        observer: Observer | None = None,
    ):
        self.env = env
        self.decider = decider
        self.config = config or NavigatorConfig()
        if self.config.evidence_format == EvidenceFormat.IMAGE:
            check_cell(self.config.costmap.size, self.config.image_cell)
        if self.config.sim_latency is not None and self.config.sim_latency <= 0:
            raise ValueError("--sim-latency must be positive")
        self.observer = observer or Observer()
        self.rng = np.random.default_rng(seed)
        self.report = EpisodeReport(
            scenario,
            decider.name,
            seed,
            evidence_format=self.config.evidence_format.value,
            sim_latency=self.config.sim_latency,
        )
        self.phase = Phase.PLANNING
        self.route: Route | None = None
        self.current: Command | None = None
        self.current_path: np.ndarray | None = None
        self.pending: Pending | None = None
        self.last_costmap: LocalCostmap | None = None
        self.hold = 0.0
        self.step_index = 0
        self.waited = 0.0
        self.instruction = self.config.instruction

    @property
    def robot(self):
        return self.env.robot

    @property
    def step(self) -> float:
        return float(self.env.step_time)

    @property
    def time(self) -> float:
        return self.step_index * self.step

    @property
    def pose(self) -> np.ndarray:
        return np.asarray(self.robot.state[:3, 0], dtype=float).copy()

    @property
    def goal(self) -> np.ndarray:
        return np.asarray(self.robot.goal[:2, 0], dtype=float).copy()

    def run(self) -> EpisodeReport:
        self._setup()
        report = self.report
        report.straight_line = float(np.linalg.norm(self.goal - self.pose[:2]))
        report.trajectory.append(_rounded(self.pose[:2]))
        self.route = self.planner.plan(self.pose[:2], self.goal)
        if self.route is None:
            report.outcome = Outcome.UNREACHABLE.value
            self.phase = Phase.FINISHED
            self.observer.on_end(self)
            return report
        report.route_length = self.route.length
        report.time_limit = (
            self.config.base_time_limit
            + self.config.time_limit_factor * self.route.length / self.speed
        )
        self.phase = Phase.WARM_UP
        self.hold = self._calibrate_hold()
        report.hold = self.hold
        self.observer.on_start(self)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jevnav-decider") as worker:
            self._drive(worker)
        report.sim_time = self.time
        self.phase = Phase.FINISHED
        self.observer.on_end(self)
        return report

    def _setup(self) -> None:
        robot = self.robot
        if getattr(robot, "lidar", None) is None:
            raise ValueError("The robot needs a lidar2d sensor: the costmap is built from it")
        robot.sensor_step()
        self.radius = float(robot.radius)
        self.speed = float(robot.vel_max[0, 0])
        self.turn_rate = float(abs(robot.vel_max[1, 0]))
        self.commands = command_set(self.speed, self.turn_rate)
        self.costmap_spec = replace(self.config.costmap, robot_radius=self.radius)
        self.limits = SafetyLimits.from_robot(robot, self.config.clearance_margin)
        self.grid = occupancy_from_env(self.env, self.config.grid_resolution)
        self.planner = RoutePlanner(
            self.grid,
            PlannerConfig(self.radius, self.config.safety_margin, self.config.comfort_distance),
        )

    def _calibrate_hold(self) -> float:
        if self.config.hold:
            return self._hold_for(0.0)
        if self.config.sim_latency:
            return self._hold_for(self.config.sim_latency)
        self.hold = 1.0
        latencies = [self._decide(self._compose(-1, pose)).latency for pose in self._warmup_poses()]
        self.report.warmup_latencies = latencies
        return self._hold_for(median(latencies))

    def _warmup_poses(self) -> list[np.ndarray]:
        turn = math.radians(DIRECTION_STEP_DEGREES)
        offsets = [
            0,
            *(sign * k for k in range(1, self.config.warmup_requests) for sign in (1, -1)),
        ]
        return [
            self.pose + np.array([0.0, 0.0, turn * offset])
            for offset in offsets[: self.config.warmup_requests]
        ]

    def _hold_for(self, latency: float) -> float:
        if self.config.hold:
            seconds = self.config.hold
        else:
            recent = [
                self._counted(record.latency)
                for record in self.report.decisions[-self.config.latency_window :]
                if record.command_id != SAFETY_STOP.id
            ]
            seconds = self.config.hold_factor * max(
                [latency, *([median(recent)] if recent else [])]
            )
        return round(max(1, self._steps(seconds)) * self.step, 6)

    def _counted(self, latency: float) -> float:
        return self.config.sim_latency or latency

    def _latency(self, decision: Decision) -> float:
        return decision.latency if decision.served_by == SAFETY else self._counted(decision.latency)

    def _await(self, future: Future) -> Decision:
        started = time.perf_counter()
        resumed = self.phase
        try:
            while True:
                try:
                    return future.result(timeout=WAIT_POLL)
                except TimeoutError:
                    self.phase = Phase.WAITING
                    self.waited = time.perf_counter() - started
                    self.observer.on_wait(self)
        finally:
            self.phase = resumed
            self.waited = 0.0

    def _steps(self, seconds: float) -> int:
        return math.ceil(seconds / self.step - 1e-9)

    @property
    def simulated(self) -> bool:
        return self.config.evidence_format in SIMULATED

    def _compose(self, index: int, pose) -> DecisionRequest:
        scan = Scan.from_robot(self.robot)
        costmap = build_costmap(self.costmap_spec, pose, scan, self.route, self.goal)
        if not self.simulated:
            return compose(
                index,
                self.step_index,
                costmap,
                self.commands,
                self.hold,
                self.rng,
                self.config.evidence_format,
                instruction=self.instruction,
            )
        verdicts = screen(self.commands, pose, self.hold, self.step, scan, self.limits)
        return compose(
            index,
            self.step_index,
            costmap,
            self.commands,
            self.hold,
            self.rng,
            self.config.evidence_format,
            candidates=[command for command in self.commands if verdicts[command.id].safe],
            route=self.route,
            step=self.step,
            verdicts=verdicts,
            instruction=self.instruction,
            image_cell=self.config.image_cell,
        )

    def _decide(self, request: DecisionRequest) -> Decision:
        if self.simulated and not request.options:
            return safety_stop()
        return self.decider.decide(request)

    def _submit(self, worker: ThreadPoolExecutor, index: int, pose) -> Pending:
        request = self._compose(index, pose)
        self.last_costmap = request.costmap
        self.observer.on_request(self, request)
        return Pending(request, worker.submit(self._decide, request))

    def _drive(self, worker: ThreadPoolExecutor) -> None:
        self.phase = Phase.AWAITING_FIRST
        self.pending = self._submit(worker, 0, self.pose)
        planned_switch = switch_step = 0
        station = 0.0
        progress = deque([(self.time, station)])
        while True:
            outcome = self._outcome()
            if outcome is not None:
                self.report.outcome = outcome.value
                return
            if self.step_index >= switch_step:
                switch_step, applied = self._resolve(worker, planned_switch)
                if applied:
                    planned_switch = switch_step
            station = self._follow(station, progress)
            previous = self.pose[:2]
            scan = Scan.from_robot(self.robot)
            self.report.min_clearance = min(
                self.report.min_clearance, scan.nearest_hit() - self.radius
            )
            self.env.step(self.current.action if self.current else [0.0, 0.0])
            self.step_index += 1
            self.report.distance_traveled += float(np.linalg.norm(self.pose[:2] - previous))
            self.report.trajectory.append(_rounded(self.pose[:2]))
            self.observer.on_step(self)

    def _resolve(self, worker: ThreadPoolExecutor, planned_switch: int) -> tuple[int, bool]:
        pending = self.pending
        decision = self._await(pending.future)
        if pending.record is None:
            pending.record = self._record(pending.request, decision)
        ready_step = pending.request.sent_step + self._steps(self._latency(decision))
        if ready_step > self.step_index:
            if self.current is not None:
                self.phase = Phase.OVERRUN
            return ready_step, False
        record = pending.record
        record.applied_at = self.time
        record.overrun = (
            max(0, self.step_index - planned_switch) * self.step if self.current else 0.0
        )
        if decision.served_by != SAFETY:
            self.hold = self._hold_for(self._latency(decision))
        record.hold = self.hold
        self.current = self._validated(pending.request, decision, record)
        self.current_path = rollout(
            self.pose, self.current.linear, self.current.angular, self.hold, self.step
        )
        self.phase = Phase.EXECUTING
        self.observer.on_decision(self, record, pending.request)
        self.pending = self._submit(worker, record.index + 1, self.current_path[-1])
        return self.step_index + round(self.hold / self.step), True

    def _validated(
        self, request: DecisionRequest, decision: Decision, record: DecisionRecord
    ) -> Command:
        if not self.simulated:
            record.executed = decision.command_id
            return request.command(decision.command_id)
        command, record.fallbacks = first_safe(
            decision.ranked() if request.options else (),
            request.options,
            self.pose,
            self.hold,
            self.step,
            Scan.from_robot(self.robot),
            self.limits,
        )
        record.executed = command.id
        return command

    def _record(self, request: DecisionRequest, decision: Decision) -> DecisionRecord:
        if decision.served_by != SAFETY:
            self.report.served_by = decision.served_by
        record = DecisionRecord(
            index=request.index,
            sent_at=request.sent_step * self.step,
            applied_at=None,
            command_id=decision.command_id,
            probabilities=decision.probabilities,
            latency=decision.latency,
            confidence=decision.confidence,
            input_tokens=decision.input_tokens,
            overrun=0.0,
            option_order=[command.id for command in request.options],
            evidence=request.evidence,
            predicted_pose=[round(float(value), 3) for value in request.costmap.pose],
            instructions=request.instructions,
            ranking=list(decision.ranked()) if request.options else [],
            removed={
                command_id: verdict.reason
                for command_id, verdict in request.verdicts.items()
                if not verdict.safe
            },
            option_texts=dict(request.descriptions),
            instruction=request.instruction,
            image_png=request.images[0] if request.images else None,
        )
        self.report.decisions.append(record)
        return record

    def _outcome(self) -> Outcome | None:
        if self.robot.arrive:
            return Outcome.ARRIVED
        if self.robot.collision:
            return Outcome.COLLIDED
        if self.time >= self.report.time_limit - 1e-9:
            return Outcome.TIMED_OUT
        return None

    def _follow(self, station: float, progress: deque) -> float:
        reach = self.speed * self.hold * 2 + 1.0
        station, offset = self.route.project(
            self.pose[:2], lower=station - 1.0, upper=station + reach
        )
        progress.append((self.time, station))
        if offset > self.config.replan_offset or self._stalled(progress):
            self._replan()
            station, _ = self.route.project(self.pose[:2])
            progress.clear()
            progress.append((self.time, station))
        return station

    def _stalled(self, progress: deque) -> bool:
        window = self.config.stall_window
        now, latest = progress[-1]
        while len(progress) >= 2 and now - progress[1][0] >= window:
            progress.popleft()
        oldest_time, oldest = progress[0]
        return now - oldest_time >= window and latest - oldest < self.config.stall_progress

    def _replan(self) -> None:
        self.report.replans += 1
        scan = Scan.from_robot(self.robot)
        fresh = self.planner.plan(self.pose[:2], self.goal, extra_obstacles=scan.points[scan.hits])
        if fresh is None:
            fresh = self.planner.plan(self.pose[:2], self.goal)
        if fresh is not None:
            self.route = fresh


def _rounded(position) -> list[float]:
    return [round(float(position[0]), 3), round(float(position[1]), 3)]
