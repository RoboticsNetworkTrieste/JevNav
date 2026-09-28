import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .deciders import (
    DEFAULT_JEV_MODEL,
    DEFAULT_JEV_URL,
    HeuristicDecider,
    JevDecider,
    JevUnavailable,
)
from .navigator import Navigator, NavigatorConfig
from .prompt import EvidenceFormat
from .report import aggregate
from .scenario import available_scenarios, open_env, resolve_scenario

CLOSE_PAUSE = 0.01
START_RIZZO_HINT = (
    "Start Rizzo Flow first, from its checkout:\n"
    "  uv run rizzo serve            (add --vision for --evidence image)"
)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except JevUnavailable as error:
        print(f"jevnav: {error}\n{START_RIZZO_HINT}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as error:
        print(f"jevnav: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevnav",
        description="A Jev-style System One model drives an ir-sim robot to its goal.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one episode")
    run.add_argument("scenario", nargs="?", default="open", help="bundled name or YAML path")
    run.add_argument("--decider", choices=["jev", "heuristic"], default="jev")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--render", action="store_true", help="show ir-sim with the decision sidebar")
    run.add_argument(
        "--speed", type=float, default=1.0, help="playback speed of --render, 1 = real time"
    )
    run.add_argument("--save-gif", type=Path, help="write an animated GIF of the run")
    run.add_argument("--save-frame", type=Path, help="write the final frame as PNG")
    run.add_argument("--report", type=Path, help="write the episode report as JSON")
    run.add_argument(
        "--trace", type=Path, help="write every decision: costmap, options, probabilities"
    )
    _decision_options(run)
    run.set_defaults(handler=_run)

    bench = commands.add_parser("bench", help="compare deciders over scenarios and seeds")
    bench.add_argument("scenarios", nargs="*", help="default: every bundled scenario")
    bench.add_argument(
        "--deciders", nargs="+", choices=["jev", "heuristic"], default=["heuristic", "jev"]
    )
    bench.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    bench.add_argument("--output", type=Path, help="write every report and the summary as JSON")
    _decision_options(bench)
    bench.set_defaults(handler=_bench)

    check = commands.add_parser("check", help="check the Jev-compatible server")
    _jev_options(check)
    check.set_defaults(handler=_check)

    scenarios = commands.add_parser("scenarios", help="list bundled scenarios")
    scenarios.set_defaults(handler=lambda _: print("\n".join(available_scenarios())) or 0)
    return parser


def _jev_options(parser) -> None:
    parser.add_argument("--jev-url", default=os.environ.get("JEVNAV_JEV_URL", DEFAULT_JEV_URL))
    parser.add_argument("--model", default=os.environ.get("JEVNAV_JEV_MODEL", DEFAULT_JEV_MODEL))
    parser.add_argument(
        "--api-key-env", default="JEVNAV_JEV_API_KEY", help="variable holding a Bearer key"
    )
    parser.add_argument("--timeout", type=float, default=60.0)


def _decision_options(parser) -> None:
    _jev_options(parser)
    parser.add_argument(
        "--hold", type=float, help="fixed command hold in seconds; default: 1.2 x latency"
    )
    parser.add_argument("--hold-factor", type=float, default=NavigatorConfig.hold_factor)
    parser.add_argument(
        "--evidence",
        choices=[value.value for value in EvidenceFormat],
        default=EvidenceFormat.GRID.value,
        help="what Rizzo reads: the ASCII costmap, its coordinates, each command's simulation, "
        "or that simulation with the costmap as an image (needs `rizzo serve --vision`)",
    )
    parser.add_argument(
        "--image-cell",
        type=int,
        default=NavigatorConfig.image_cell,
        help="image evidence: pixels per costmap cell, a multiple of 32 giving 1024-4096 "
        "Qwen3-VL image tokens: 64 (1344 px, 1764 tokens) or 96",
    )
    parser.add_argument(
        "--sim-latency",
        type=float,
        help="fake timing: every answer counts as this many simulated seconds, and the "
        "simulation pauses until it arrives; default: measured latency, real time",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=NavigatorConfig.clearance_margin,
        help="simulation only: minimum clearance in meters for a command to be offered",
    )
    parser.add_argument(
        "--instruction", help="operator instruction added to every request; editable in --render"
    )
    parser.add_argument(
        "--heuristic-latency", type=float, help="simulated latency of the heuristic decider"
    )


def _decider(name: str, args):
    if name == "heuristic":
        return HeuristicDecider(latency=args.heuristic_latency)
    decider = JevDecider(args.jev_url, args.model, os.environ.get(args.api_key_env), args.timeout)
    decider.served_models()
    return decider


def _config(args) -> NavigatorConfig:
    return NavigatorConfig(
        hold=args.hold,
        hold_factor=args.hold_factor,
        evidence_format=EvidenceFormat(args.evidence),
        clearance_margin=args.clearance,
        instruction=args.instruction,
        image_cell=args.image_cell,
        sim_latency=args.sim_latency,
    )


def _run(args) -> int:
    path = resolve_scenario(args.scenario)
    decider = _decider(args.decider, args)
    visual = args.render or args.save_gif or args.save_frame
    env = open_env(path, display=args.render, figure=bool(visual), seed=args.seed)
    visualizer = None
    if visual:
        from .visualizer import Visualizer

        visualizer = Visualizer(
            env,
            interactive=args.render,
            speed=args.speed,
            gif_path=args.save_gif,
            frame_path=args.save_frame,
        )
    navigator = Navigator(
        env, decider, _config(args), scenario=path.stem, seed=args.seed, observer=visualizer
    )
    try:
        report = navigator.run()
    except BaseException:
        _close(env)
        raise
    print(report.line())
    if args.report:
        _write_json(args.report, report.to_dict())
    if args.trace:
        _write_json(args.trace, report.trace())
    if args.save_gif:
        print(f"wrote {args.save_gif}")
    if args.save_frame:
        print(f"wrote {args.save_frame}")
    if visualizer is not None and args.render:
        print("close the window to exit")
        visualizer.hold_open()
    _close(env)
    return 0 if report.outcome == "arrived" else 1


def _bench(args) -> int:
    paths = [resolve_scenario(name) for name in (args.scenarios or available_scenarios())]
    deciders = sorted(
        (_decider(name, args) for name in args.deciders), key=lambda decider: decider.name != "jev"
    )
    config = _config(args)
    reports = []
    for path in paths:
        for decider in deciders:
            _match_model_latency(decider, reports, args)
            for seed in args.seeds:
                env = open_env(path, seed=seed)
                try:
                    report = Navigator(env, decider, config, scenario=path.stem, seed=seed).run()
                finally:
                    _close(env)
                reports.append(report)
                print(report.line(), flush=True)
    rows = aggregate(reports)
    print()
    _print_table(rows)
    if args.output:
        _write_json(
            args.output, {"summary": rows, "episodes": [report.to_dict() for report in reports]}
        )
    return 0


def _match_model_latency(decider, reports, args) -> None:
    if not isinstance(decider, HeuristicDecider) or args.heuristic_latency is not None:
        return
    observed = [
        value for report in reports if report.decider == "jev" for value in report.latencies
    ]
    if observed:
        decider.latency = float(np.median(observed))
        print(f"heuristic runs with the model's median latency, {decider.latency:.2f} s")


def _close(env) -> None:
    env.end(ending_time=CLOSE_PAUSE)


def _check(args) -> int:
    decider = JevDecider(args.jev_url, args.model, os.environ.get(args.api_key_env), args.timeout)
    print(f"{args.jev_url} serves: {', '.join(decider.served_models())}")
    return 0


def _print_table(rows) -> None:
    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
