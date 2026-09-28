# CLAUDE.md

JEVNAV: Rizzo Flow (a local Jev-style System One model, `external/rizzo-flow`) drives a
differential-drive robot in ir-sim (`external/ir-sim`) from one lidar costmap per decision.

## Working method: SPECS² (visual shared model)

- The shared model lives in `.specs/`: `architecture.md` plus one file per feature, each with
  one primary Mermaid view (Flow, Interaction or State). Read it before changing code.
- For every change: update the smallest affected view first, show the user a short **Model
  Delta** (added / changed / removed), implement freely, then re-sync the model with what was
  actually built.
- The user reviews the docs before large implementation steps: ask before implementing when
  the model changed materially.
- Source code is comment-free by design: no explanatory comments or docstrings. Intent goes
  in names, types, tests and `.specs/`.
- `command-set.md`'s table is generated from `src/jevnav/commands.py`; keep them identical.

## Environment (macOS)

- Everything heavy lives outside the repository in `~/.cache/jevnav`: venvs (`venvs/jevnav`,
  `venvs/rizzo-flow`), Rizzo weights and the llama.cpp Metal runtime (`rizzo/`). Rizzo Flow's
  branch `vlm-costmap-image` (checked out in `external/rizzo-flow`) adds Qwen3-VL-4B-Instruct
  Q8_0 + its mmproj in `rizzo/models/Qwen3-VL-4B-Instruct-GGUF/`, served by
  `rizzo serve --vision` through the runtime's `llama-server`.
- `uv` is in `~/.local/bin`. Always `export UV_PROJECT_ENVIRONMENT=~/.cache/jevnav/venvs/jevnav`.

## Commands

```bash
export PATH="$HOME/.local/bin:$PATH" UV_PROJECT_ENVIRONMENT=~/.cache/jevnav/venvs/jevnav
uv run pytest -q                          # no model needed: stubs + heuristic
uv run ruff format src tests && uv run ruff check src tests
cd ~/.cache/jevnav/rizzo && ~/.cache/jevnav/venvs/rizzo-flow/bin/rizzo serve   # port 8017
uv run jevnav run open --render           # Rizzo drives, window with decision sidebar
uv run jevnav run open --render --evidence text   # Rizzo reads coordinates instead of the grid
uv run jevnav run slalom --decider heuristic --save-gif runs/slalom.gif
uv run jevnav bench --seeds 0 1 2 --output runs/bench.json
uv run jevnav run slalom --render --evidence simulation   # Snake logic: Rizzo picks among simulated commands
uv run jevnav run slalom --render --evidence simulation --instruction "Stay far from obstacles."
cd ~/.cache/jevnav/rizzo && ~/.cache/jevnav/venvs/rizzo-flow/bin/rizzo serve --vision   # Qwen3-VL-4B, port 8017
uv run jevnav run slalom --render --evidence image --sim-latency 1   # costmap picture; each answer counts as 1 s, the sim waits
```

- The user runs the software and experiments (`jevnav run`, `bench`) themselves. Only unit tests and linting run without asking.
- One model process at a time (about 6 GB of GPU memory). Stop the server before timing
  anything else on the GPU.
- Rizzo latency drifts upward under sustained load on this fanless Mac; the adaptive hold
  handles it. Compare timings only between runs from a similar thermal state.
