# CLAUDE.md

JEVNAV: CLM-8B (a Contrastive Language Model used as a local Jev-style System One model,
`external/clm`) drives a differential-drive robot in ir-sim (`external/ir-sim`) from one lidar
costmap per decision.

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
  `venvs/clm`) and `clm/`: the Qwen3-8B Q8_0 encoder (`clm/models/Qwen3-8B-GGUF/`), CLM's
  head (`clm/heads/CLM_v0.1-8B.pt`) and the llama.cpp b11081 Metal runtime
  (`clm/runtime/llama-b11081/`).
- CLM ships for vLLM (Linux + NVIDIA). Here `external/clm` is installed into `venvs/clm` with
  `--no-deps` (no vLLM), llama.cpp's `llama-server --embeddings --pooling last` serves the
  encoder on :8090, and `clm-serve` runs the heads on the CPU on :8700.
  `scripts/serve-clm.sh` starts both, bound to 127.0.0.1.
- `--evidence image` is dormant: CLM v0.1 reads text only, so the CLI refuses it with
  `--decider jev` (`.specs/features/image-evidence.md`).
- `uv` is in `~/.local/bin`. Always `export UV_PROJECT_ENVIRONMENT=~/.cache/jevnav/venvs/jevnav`.

## Commands

```bash
export PATH="$HOME/.local/bin:$PATH" UV_PROJECT_ENVIRONMENT=~/.cache/jevnav/venvs/jevnav
uv run pytest -q                          # no model needed: stubs + heuristic
uv run ruff format src tests && uv run ruff check src tests
scripts/serve-clm.sh                      # Qwen3-8B encoder :8090 + CLM :8700, Ctrl-C stops both
uv run jevnav check                       # is CLM answering?
uv run jevnav run open --render           # CLM drives, window with decision sidebar
uv run jevnav run open --render --evidence text   # CLM reads coordinates instead of the grid
uv run jevnav run slalom --decider heuristic --save-gif runs/slalom.gif
uv run jevnav bench --seeds 0 1 2 --output runs/bench.json
uv run jevnav run slalom --render --evidence simulation --sim-latency 1   # Snake logic; 7-13 s per CLM answer, the sim waits
uv run jevnav run slalom --render --evidence simulation --sim-latency 1 --instruction "Stay far from obstacles."
uv run jevnav run slalom --decider heuristic --render --evidence image    # dormant mode, heuristic only
```

- The user runs the software and experiments (`jevnav run`, `bench`) themselves. Only unit tests and linting run without asking.
- One model process at a time: the encoder takes about 9.5 GB of the Mac's 16 GB. Stop
  `serve-clm.sh` before timing anything else, and before running the tests (they slow down
  about 7× with it loaded).
- Latency drifts upward under sustained load on this fanless Mac; the adaptive hold
  handles it. Compare timings only between runs from a similar thermal state.
