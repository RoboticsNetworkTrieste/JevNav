# JEVNAV

**A contrastive language model drives a robot, one probability distribution at a time.**

JEVNAV lets [CLM-8B](https://github.com/Contrastive-LM/CLM), a Contrastive Language Model used
as a local, Jev-style "System One" model, drive a differential-drive robot to its goal in the
[ir-sim](https://github.com/hanruihua/ir-sim) 2D simulator. At every decision the model embeds
what the lidar sees and each of 24 fixed motion commands, and returns a probability for each
command from how well it matches the situation: it never generates text.

![JEVNAV on the slalom scenario: the ir-sim view with the robot's path, and a sidebar with the probability of each command, the costmap the model saw and the decision log](docs/slalom-operator-instruction.gif)

*Slalom scenario, simulation evidence, replayed at 8× speed. Recorded with the previous model
server, before the switch to CLM. After ten decisions the operator types "prefer go backward";
the evidence and the options stay the same, only the words of the question change.*

## How it works

1. **Plan once.** A clearance-aware A* on the scenario's static map gives the global route.
2. **Look.** Before each command starts, the latest lidar scan (180 beams, 5 m) becomes a
   21 × 21 local costmap with 0.2 m cells. It is centred on the pose the robot will have when
   the command starts, with the route and the goal drawn in.
3. **Ask.** JEVNAV sends the evidence and the 24 commands, shuffled, to CLM's Jev-compatible
   `POST /v1/systemone`. CLM embeds the evidence with the question once and every command's
   text on its own (cached after the first time) with a frozen Qwen3-8B encoder and two small
   trained heads, and returns the softmax of their scaled cosine similarities.
4. **Act.** The chosen command is held for H seconds, adaptive at 1.2 × the model's measured
   latency, while the next decision is already being computed in the background.
5. **Recover.** The route is replanned when the robot drifts more than 1 m from it or stops
   making progress.

The 24 commands are travel directions 15° apart: straight ahead and back at 0.25 m/s, rotate
in place at 45°/s, and forward or backward arcs turning at 7.5 to 37.5°/s
([full table](.specs/features/command-set.md)).

### What the model reads: `--evidence`

| Mode | CLM reads | Safety net |
| --- | --- | --- |
| `grid` (default) | the costmap as an ASCII grid with a legend | none, on purpose: the most probable command is executed as is, and a collision ends the episode |
| `text` | the same costmap as coordinates in the robot frame: pose, goal, route and obstacle points | none |
| `simulation` | "Snake logic": every command is simulated for the hold, and unsafe ones (collision, clearance under 0.05 m, velocity limits) are removed. Each remaining option says where it would leave the robot: route still to go, distance off the route, straight-line distance to the goal, closest obstacle | the ranking is rechecked from the actual pose at the end of the hold; the next safe command, or a stop, replaces an unsafe one |
| `image` | dormant: `simulation` plus the costmap as a PNG, for a vision model. CLM v0.1 reads text only, so JEVNAV refuses this mode with `--decider jev`; the heuristic runs it | as `simulation` |

A classical **heuristic decider** (`--decider heuristic`) reads the same costmap and serves as
the baseline. It needs no model.

With `--render`, an **operator instruction** box and presets (keep clear, no reverse, hurry,
prefer left, prefer right) add a sentence to every new question while the robot drives;
`--instruction "…"` does the same from the command line.

## Requirements

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/).
- For the model: about 10 GB of free memory for CLM's Qwen3-8B encoder (Q8_0) and its heads.
  JEVNAV serves the encoder with [llama.cpp](https://github.com/ggml-org/llama.cpp) and was
  developed and measured on an Apple M5 Mac with 16 GB (Metal). On Linux with an NVIDIA GPU,
  CLM's own vLLM encoder works too: point `clm-serve --emb-url` at it.
- The heuristic decider and the tests need neither.

## Install

```bash
git clone https://github.com/RoboticsNetworkTrieste/JevNav.git
cd JevNav

git clone https://github.com/hanruihua/ir-sim.git external/ir-sim
git -C external/ir-sim checkout dd7f1e5

uv sync
uv run pytest -q
```

ir-sim is installed in editable mode from `external/ir-sim`; `dd7f1e5` is the tested commit.

### Model server

CLM runs as two local processes with their own environment: llama.cpp's `llama-server` turns
texts into Qwen3-8B embeddings, and `clm-serve` runs CLM's heads and answers JEVNAV. CLM lists
vLLM as a dependency, which needs Linux and an NVIDIA GPU, so it is installed without its
dependencies and these are added by hand.

```bash
git clone https://github.com/Contrastive-LM/CLM.git external/clm
git -C external/clm checkout bb42c6c

uv venv --python 3.12 ~/.cache/jevnav/venvs/clm
uv pip install --python ~/.cache/jevnav/venvs/clm/bin/python --no-deps -e external/clm
uv pip install --python ~/.cache/jevnav/venvs/clm/bin/python numpy requests torch fastapi uvicorn

mkdir -p ~/.cache/jevnav/clm/{runtime,models/Qwen3-8B-GGUF,heads}
curl -L https://github.com/ggml-org/llama.cpp/releases/download/b11081/llama-b11081-bin-macos-arm64.tar.gz \
  | tar -xz -C ~/.cache/jevnav/clm/runtime
curl -L -o ~/.cache/jevnav/clm/models/Qwen3-8B-GGUF/Qwen3-8B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf      # 8.7 GB
curl -L -o ~/.cache/jevnav/clm/heads/CLM_v0.1-8B.pt \
  https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/main/CLM_v0.1-8B.pt  # 75 MB

scripts/serve-clm.sh      # encoder on 127.0.0.1:8090, CLM on 127.0.0.1:8700
```

`scripts/serve-clm.sh` starts the encoder, waits until it is healthy, then runs `clm-serve`
in the foreground; Ctrl-C stops both. CLM's playground is at http://127.0.0.1:8700/. Run one
model server at a time.

## Run

```bash
uv run jevnav scenarios                                   # open, slalom, crossing
uv run jevnav run slalom --decider heuristic --render     # no model needed
uv run jevnav check                                       # is CLM answering?
uv run jevnav run open --render                           # CLM drives, grid evidence
uv run jevnav run open --render --evidence text
uv run jevnav run slalom --render --evidence simulation --sim-latency 1
uv run jevnav run slalom --render --evidence simulation --sim-latency 1 --instruction "Stay far from obstacles."
uv run jevnav bench --seeds 0 1 2 --output runs/bench.json
```

A scenario is a bundled name or a path to an ir-sim YAML file. `bench` runs every scenario ×
decider × seed; the heuristic gets CLM's median latency, so both see the same holds.

| Option | Effect |
| --- | --- |
| `--render`, `--speed 2` | live window: the ir-sim view plus a sidebar with a probability compass, the costmap the model saw and the decision log; playback speed |
| `--save-gif PATH`, `--save-frame PATH` | an animated GIF of the run, or its last frame, headless |
| `--report PATH` | the episode report: outcome, time, path efficiency, clearance, latency, overruns |
| `--trace PATH` | every decision: evidence, option order, the 24 probabilities |
| `--hold H` | a fixed hold instead of 1.2 × latency |
| `--sim-latency S` | fake timing for slow models: every answer counts as S simulated seconds, and the simulation pauses while the model thinks |
| `--clearance M` | `simulation` and `image`: the minimum clearance for a command to be offered |
| `--jev-url`, `--model`, `--api-key-env` | any Jev-compatible `/v1/systemone` endpoint |

On the Apple M5 with warm servers, a `grid` or `text` decision takes about 1.1 s: CLM embeds
only the new state, because the 24 command texts are cached. A `simulation` decision takes
7–13 s, because every option carries new numbers and must be embedded again; that is what
`--sim-latency` is for.

## Limitations

- CLM sees one snapshot per decision, never a history, so moving obstacles look static.
- `grid` and `text` have no safety net by design: they measure what the model does alone.
- CLM judges every option on its own, never side by side, and its training data holds no
  costmaps. In a first replay of recorded requests it chose the same command for every grid
  map, and in `simulation` it picked the option with the least route still to go in 4 of 16
  decisions ([details](.specs/features/model-decision.md)). `bench` measures it against the
  heuristic baseline.
- The encoder runs as Q8_0 through llama.cpp, while CLM's head was trained on bf16 vLLM
  embeddings. On CLM's README example the answers point the same way but are sharper.
- `image` evidence waits for a CLM that reads images.
- Only macOS on Apple Silicon has been tested.

## Design

The design lives in [`.specs/`](.specs/architecture.md) as a shared visual model:
[`architecture.md`](.specs/architecture.md) plus one file per feature, each built around one
primary view.

| Feature | View |
| --- | --- |
| [Navigation episode](.specs/features/navigation-episode.md) | State |
| [Global route](.specs/features/global-route.md) | Flow |
| [Local costmap](.specs/features/local-costmap.md) | Flow |
| [Command set](.specs/features/command-set.md) | Direction chart + table, generated from `commands.py` |
| [Model decision](.specs/features/model-decision.md) | Interaction |
| [Text evidence](.specs/features/text-evidence.md) | Flow |
| [Simulation decision](.specs/features/simulation-decision.md) | Flow |
| [Image evidence](.specs/features/image-evidence.md) (dormant) | Interaction |
| [Operator instruction](.specs/features/operator-instruction.md) | Interaction |

Each change starts from the smallest affected view, and the view is re-synced with what was
built. The source code has no comments or docstrings on purpose: intent lives in names,
types, tests and the specs.

```text
src/jevnav/            navigator, costmap, route planner, safety, simulation, prompt,
                       deciders, visualizer, CLI
src/jevnav/scenarios/  open, slalom, crossing (ir-sim YAML)
scripts/serve-clm.sh   starts the Qwen3-8B encoder and CLM
tests/                 unit tests with stub deciders, no model needed
.specs/                the design
docs/                  README media
external/              ir-sim and CLM checkouts, not tracked
```

## Development

```bash
uv run pytest -q
uv run ruff format src tests && uv run ruff check src tests
```

## Acknowledgements

- [CLM](https://github.com/Contrastive-LM/CLM) by Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon
  Saad-Falcon, Marco Pavone, Christopher Ré and Azalia Mirhoseini (Apache-2.0, code and
  CLM-8B weights), the model and its server.
- [ir-sim](https://github.com/hanruihua/ir-sim) by Ruihua Han (MIT), the simulator.
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by the Qwen team (Apache-2.0), CLM's
  encoder, and [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), which serves it.

"Jev" and "TypeSafe" are names of TypeSafe's products; JEVNAV is independent and not
affiliated.

## License

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).
