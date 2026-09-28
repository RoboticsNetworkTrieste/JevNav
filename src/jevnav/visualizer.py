import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.widgets import Button, TextBox

from .costmap import GOAL_SYMBOL, ROBOT_SYMBOL, ROUTE_SYMBOL, LocalCostmap
from .kinematics import rollout
from .navigator import Navigator, Observer, Phase
from .palette import (
    BAR,
    BAR_CHOSEN,
    BASELINE,
    CELL_COLORS,
    COMMAND,
    GRIDLINE,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    ROUTE,
    SURFACE,
)
from .prompt import DecisionRequest
from .report import DecisionRecord

CELL_LABELS = {
    "#": "obstacle",
    "x": "too close",
    "+": "near obstacle",
    ".": "free",
    "?": "unknown",
    ROUTE_SYMBOL: "route",
    GOAL_SYMBOL: "goal",
    ROBOT_SYMBOL: "robot",
}
LOG_ROWS = 11
PRESETS = {
    "Keep clear": "Stay as far from obstacles as you can, even if it costs some progress.",
    "No reverse": "Do not drive backward unless no forward command makes progress.",
    "Hurry": (
        "Make the most progress along the route; clearance matters only when progress is equal."
    ),
    "Prefer left": "When commands make similar progress, prefer the ones that curve left.",
    "Prefer right": "When commands make similar progress, prefer the ones that curve right.",
    "Clear": "",
}
CONTROLS_LEFT = 0.04
CONTROLS_WIDTH = 0.53
LABEL_WIDTH = 0.055
BUTTON_GAP = 0.004
WORLD_BELOW_CONTROLS = [0.04, 0.05, 0.53, 0.80]
STATUS_CHARACTERS = 72
HOLD_OPEN_POLL = 0.1
COSTMAP_TITLES = {
    "grid": "Costmap the model saw",
    "text": "Costmap the model read as text",
    "simulation": "Costmap behind the simulation",
    "image": "Costmap image the model saw",
}


class Visualizer(Observer):
    def __init__(
        self,
        env,
        *,
        interactive: bool = True,
        speed: float = 1.0,
        gif_path: Path | None = None,
        gif_every: int = 2,
        frame_path: Path | None = None,
    ):
        self.env = env
        self.interactive = interactive
        self.speed = speed
        self.gif_path = gif_path
        self.gif_every = max(1, gif_every)
        self.frame_path = frame_path
        self.frames: list = []
        self.records: list[DecisionRecord] = []
        self.shown: DecisionRequest | None = None
        self.frame_due: float | None = None
        self.figure = env._env_plot.fig
        self.world_ax = env._env_plot.ax

    def on_start(self, navigator: Navigator) -> None:
        self.navigator = navigator
        self._layout(navigator)
        if self.interactive:
            self._layout_controls(navigator)
            self._mute_simulator_keys()
        self.frame_due = time.perf_counter()
        self._update_status(navigator)
        self.env.render(0.001)
        if self.interactive:
            self._stop_raising_the_window()

    def on_request(self, navigator: Navigator, request: DecisionRequest) -> None:
        pose = request.costmap.pose
        self.next_pose.set_data([pose[0]], [pose[1]])

    def on_decision(
        self, navigator: Navigator, record: DecisionRecord, request: DecisionRequest
    ) -> None:
        self.records.append(record)
        self.shown = request
        self._update_compass(navigator, record)
        self._update_costmap(navigator, record, request.costmap)
        self._update_log()

    def on_step(self, navigator: Navigator) -> None:
        self._update_world(navigator)
        self._update_status(navigator)
        self.env.render(0.001)
        if self.gif_path and navigator.step_index % self.gif_every == 0:
            self._capture()
        self._pace(navigator)

    def on_wait(self, navigator: Navigator) -> None:
        self._update_status(navigator)
        if self.interactive:
            self.figure.canvas.draw_idle()
            plt.pause(0.001)

    def on_end(self, navigator: Navigator) -> None:
        if not hasattr(self, "status"):
            self._layout(navigator)
        self._update_world(navigator)
        self._update_status(navigator)
        self.env.render(0.001)
        self.figure.canvas.draw()
        if self.gif_path:
            self._capture()
            self._write_gif()
        if self.frame_path:
            self.frame_path.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(self.frame_path, dpi=110, facecolor=SURFACE)

    def hold_open(self) -> None:
        if not self.interactive:
            return
        while plt.fignum_exists(self.figure.number):
            self.figure.canvas.start_event_loop(HOLD_OPEN_POLL)

    @staticmethod
    def _stop_raising_the_window() -> None:
        plt.rcParams["figure.raise_window"] = False

    def _layout(self, navigator: Navigator) -> None:
        figure = self.figure
        figure.set_size_inches(16, 9, forward=True)
        figure.set_facecolor(SURFACE)
        manager = getattr(figure.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(
                f"JEVNAV · {navigator.report.scenario} · {navigator.decider.name} · "
                f"{navigator.report.evidence_format} evidence"
            )
        self.world_ax.set_position([0.04, 0.07, 0.53, 0.86])
        (self.route_line,) = self.world_ax.plot(
            [], [], "--", color=ROUTE, linewidth=1.3, zorder=4, label="global route"
        )
        (self.command_line,) = self.world_ax.plot(
            [], [], "-", color=COMMAND, linewidth=2.2, zorder=5, label="command being held"
        )
        (self.next_pose,) = self.world_ax.plot(
            [],
            [],
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=COMMAND,
            markeredgewidth=2,
            linestyle="none",
            zorder=6,
            label="pose where the next decision starts",
        )
        self.world_ax.legend(
            loc="upper left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRIDLINE
        )
        self.status = figure.text(
            0.605, 0.965, "", va="top", ha="left", fontsize=10, color=INK, linespacing=1.5
        )
        self._layout_compass(navigator)
        self._layout_costmap(navigator)
        self.log = figure.text(
            0.605,
            0.30,
            "",
            va="top",
            ha="left",
            fontsize=8.5,
            family="monospace",
            color=INK_SECONDARY,
        )
        figure.text(
            0.605, 0.325, "Decision log (newest first)", fontsize=10, color=INK, va="bottom"
        )

    def _layout_compass(self, navigator: Navigator) -> None:
        ax = self.figure.add_axes([0.61, 0.40, 0.17, 0.30], projection="polar")
        ax.set_facecolor(SURFACE)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        self.command_ids = [command.id for command in navigator.commands]
        angles = np.radians([command.direction for command in navigator.commands])
        self.bars = ax.bar(
            angles,
            np.zeros(len(angles)),
            width=np.radians(15) * 0.8,
            color=BAR,
            edgecolor=SURFACE,
            linewidth=1,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1"], fontsize=7, color=INK_MUTED)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["F", "L", "B", "R"], fontsize=9, color=INK_SECONDARY)
        ax.grid(color=GRIDLINE, linewidth=0.6)
        ax.spines["polar"].set_color(BASELINE)
        ax.set_title("Probability of each command", fontsize=10, color=INK, pad=14)
        self.chosen_label = ax.text(
            0, 0, "", fontsize=9, color=INK, ha="center", va="center", fontweight="bold"
        )
        self.compass = ax

    def _layout_costmap(self, navigator: Navigator) -> None:
        ax = self.figure.add_axes([0.80, 0.40, 0.18, 0.30])
        size = navigator.costmap_spec.size
        self.costmap_image = ax.imshow(np.ones((size, size, 3)), interpolation="nearest")
        self.unknown_dots = ax.scatter([], [], s=1.5, color=BASELINE, zorder=2)
        (self.costmap_path,) = ax.plot([], [], color=COMMAND, linewidth=2, zorder=3)
        ax.plot(
            [size // 2],
            [size // 2],
            marker="^",
            markersize=8,
            color=COMMAND,
            markeredgecolor=SURFACE,
            zorder=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(BASELINE)
        self.costmap_title = COSTMAP_TITLES.get(
            navigator.report.evidence_format, "Costmap behind the evidence"
        )
        ax.set_title(self.costmap_title, fontsize=10, color=INK, pad=6)
        handles = [
            Patch(facecolor=CELL_COLORS[symbol], edgecolor=BASELINE, label=label)
            for symbol, label in CELL_LABELS.items()
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            fontsize=7.5,
            frameon=False,
            handlelength=1.2,
        )
        self.costmap_ax = ax

    def _update_world(self, navigator: Navigator) -> None:
        if navigator.route is not None:
            self.route_line.set_data(navigator.route.points[:, 0], navigator.route.points[:, 1])
        if navigator.current_path is not None:
            self.command_line.set_data(navigator.current_path[:, 0], navigator.current_path[:, 1])

    def _layout_controls(self, navigator: Navigator) -> None:
        figure = self.figure
        self.world_ax.set_position(WORLD_BELOW_CONTROLS)
        figure.text(
            CONTROLS_LEFT,
            0.985,
            "Operator instruction · Enter applies it to the next request",
            fontsize=10,
            color=INK,
            va="top",
        )
        box = figure.add_axes(
            [CONTROLS_LEFT + LABEL_WIDTH, 0.928, CONTROLS_WIDTH - LABEL_WIDTH, 0.034]
        )
        self.textbox = TextBox(
            box,
            "Instruction ",
            initial=navigator.instruction or "",
            color=SURFACE,
            hovercolor=GRIDLINE,
            textalignment="left",
        )
        self.textbox.label.set_fontsize(9)
        self.textbox.label.set_color(INK)
        self.textbox.text_disp.set_fontsize(8.5)
        self.textbox.on_submit(self._apply_instruction)
        width = (CONTROLS_WIDTH - BUTTON_GAP * (len(PRESETS) - 1)) / len(PRESETS)
        self.buttons = []
        for position, (label, sentence) in enumerate(PRESETS.items()):
            axes = figure.add_axes(
                [CONTROLS_LEFT + position * (width + BUTTON_GAP), 0.886, width, 0.034]
            )
            button = Button(axes, label, color=SURFACE, hovercolor=GRIDLINE)
            button.label.set_fontsize(8)
            button.label.set_color(INK_SECONDARY)
            button.on_clicked(lambda _event, sentence=sentence: self.textbox.set_val(sentence))
            self.buttons.append(button)

    def _apply_instruction(self, text: str) -> None:
        self.navigator.instruction = text.strip() or None
        self._update_status(self.navigator)
        self.figure.canvas.draw_idle()

    def _mute_simulator_keys(self) -> None:
        keyboard = getattr(self.env, "keyboard", None)
        for name in ("_mpl_press_cid", "_mpl_release_cid"):
            connection = getattr(keyboard, name, None)
            if connection is not None:
                self.figure.canvas.mpl_disconnect(connection)

    def _update_status(self, navigator: Navigator) -> None:
        report = navigator.report
        served = report.served_by or navigator.decider.name
        current = navigator.current.id if navigator.current else "none, standing still"
        latest = self.records[-1] if self.records else None
        chosen = f"{current}  (p {latest.probability:.2f})" if latest else current
        latencies = report.latencies
        p50 = f"{np.median(latencies):.2f} s" if latencies else "-"
        phase = navigator.phase.value
        if navigator.phase == Phase.FINISHED:
            phase = f"finished: {report.outcome.replace('_', ' ')}"
        if navigator.phase == Phase.WAITING:
            phase = f"paused, waiting for the model {navigator.waited:4.1f} s"
        timing = (
            f"    answers count as {navigator.config.sim_latency:g} s"
            if navigator.config.sim_latency
            else ""
        )
        pending = navigator.pending.request.index if navigator.pending else "-"
        instruction = navigator.instruction or "none"
        if len(instruction) > STATUS_CHARACTERS:
            instruction = instruction[: STATUS_CHARACTERS - 1] + "…"
        if navigator.instruction and navigator.decider.name == "heuristic":
            instruction += "  (the heuristic ignores it)"
        self.status.set_text(
            f"JEVNAV · {report.scenario} · {served} · {report.evidence_format} evidence\n"
            f"time {navigator.time:5.1f} s    hold H = {navigator.hold:.1f} s{timing}    {phase}\n"
            f"command held: {chosen}\n"
            f"instruction: {instruction}\n"
            f"next decision: request #{pending} in progress\n"
            f"decisions {len(report.applied)} · latency p50 {p50} · "
            f"overruns {report.overruns} · replans {report.replans}"
        )

    def _update_compass(self, navigator: Navigator, record: DecisionRecord) -> None:
        executed = record.executed or record.command_id
        for bar, command_id in zip(self.bars, self.command_ids, strict=True):
            bar.set_height(record.probabilities.get(command_id, 0.0))
            bar.set_facecolor(BAR_CHOSEN if command_id == executed else BAR)
        chosen = next((command for command in navigator.commands if command.id == executed), None)
        probability = record.probabilities.get(executed, 0.0)
        if chosen is None:
            self.chosen_label.set_position((0.0, 0.0))
        else:
            radius = min(probability + 0.2, 1.1)
            self.chosen_label.set_position((np.radians(chosen.direction), radius))
        self.chosen_label.set_text(f"{executed}\n{probability:.2f}")
        self.compass.set_title(
            f"Probability of each command · #{record.index}", fontsize=10, color=INK, pad=14
        )

    def _update_costmap(
        self, navigator: Navigator, record: DecisionRecord, costmap: LocalCostmap
    ) -> None:
        symbols = costmap.symbols()
        palette = {symbol: to_rgb(color) for symbol, color in CELL_COLORS.items()}
        self.costmap_image.set_data(
            np.array([[palette[symbol] for symbol in row] for row in symbols])
        )
        rows, cols = np.nonzero(symbols == "?")
        self.unknown_dots.set_offsets(
            np.column_stack([cols, rows]) if len(rows) else np.empty((0, 2))
        )
        command = navigator.current
        path = rollout(
            costmap.pose, command.linear, command.angular, navigator.hold, navigator.step
        )
        cells = costmap.to_cells(path[:, :2])
        self.costmap_path.set_data(cells[:, 1], cells[:, 0])
        self.costmap_ax.set_title(
            f"{self.costmap_title} · #{record.index}", fontsize=10, color=INK, pad=6
        )

    def _update_log(self) -> None:
        header = " #    sent  applied  choice    p     latency   hold  overrun"
        lines = [header]
        for record in reversed(self.records[-LOG_ROWS:]):
            overrun = f"{record.overrun:.1f} s" if record.overrun > 0 else "-"
            hold = f"{record.hold:.1f} s" if record.hold else "-"
            lines.append(
                f"{record.index:>2}  {record.sent_at:6.1f}  {record.applied_at:6.1f}   "
                f"{record.executed or record.command_id:<6}  {record.probability:4.2f}   "
                f"{record.latency:5.2f} s  "
                f"{hold:>5}  {overrun:>6}"
            )
        self.log.set_text("\n".join(lines))

    def _pace(self, navigator: Navigator) -> None:
        if not self.interactive or self.speed <= 0 or self.frame_due is None:
            return
        now = time.perf_counter()
        self.frame_due = max(self.frame_due + navigator.step / self.speed, now)
        if self.frame_due > now:
            plt.pause(self.frame_due - now)

    def _capture(self) -> None:
        from PIL import Image

        self.figure.canvas.draw()
        rgba = np.asarray(self.figure.canvas.buffer_rgba())
        image = Image.fromarray(rgba[..., :3].copy()).reduce(2)
        self.frames.append(image.quantize(colors=128))

    def _write_gif(self) -> None:
        if not self.frames:
            return
        self.gif_path.parent.mkdir(parents=True, exist_ok=True)
        duration = int(self.gif_every * self.env.step_time * 1000 / max(self.speed, 1e-6))
        self.frames[0].save(
            self.gif_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
