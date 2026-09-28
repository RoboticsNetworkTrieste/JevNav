
# JEVNAV — Architecture

## Goal

Drive a differential-drive robot in ir-sim to its goal, avoiding collisions and following an
optimized route. Every motion decision is made by Rizzo Flow, a local Jev-style "System One"
model. Its input is the latest local costmap built from the lidar. Its output is a probability
for each of the same 24 motion commands.

With `rizzo serve --vision` (Rizzo Flow branch `vlm-costmap-image`), Rizzo's base model is
the vision-language model Qwen3-VL-4B-Instruct, and `--evidence image` sends the costmap as a
picture next to the simulated options (features/image-evidence.md).

## 1 · System Context

```mermaid
flowchart LR
    dev(["Developer"])
    jevnav["JEVNAV<br/>navigation experiment runner"]
    irsim[["ir-sim<br/>2D robot simulator · in-process library"]]
    rizzo[["Rizzo Flow server<br/>Jev-compatible decision model · localhost:8017"]]
    files[("Scenarios YAML<br/>Reports + decision traces JSON")]

    dev -->|"jevnav run / bench / check"| jevnav
    jevnav <-->|"command v, ω  ·  pose, lidar scan, arrive / collision flags"| irsim
    jevnav <-->|"POST /v1/systemone: evidence (costmap text, simulated results, costmap image) + options → probabilities"| rizzo
    jevnav <-->|"read scenarios · write reports"| files
```

## 2 · Containers and deployment

```mermaid
flowchart TB
    subgraph mac["Developer Mac · Apple M5"]
        subgraph repo["JEVNAV/ repository — source only"]
            src["src/jevnav + bundled scenarios"]
            specs[".specs/ — this shared model"]
            ext["external/ir-sim · external/rizzo-flow"]
        end
        subgraph cache["~/.cache/jevnav — outside the repository"]
            venv1["venvs/jevnav<br/>Python 3.12 · ir-sim editable"]
            venv2["venvs/rizzo-flow<br/>Python 3.12"]
            weights["rizzo/models<br/>Spark-X2.5-4B-Q8_0.gguf · 4.4 GB<br/>Qwen3VL-4B-Instruct-Q8_0.gguf · 4.3 GB<br/>+ mmproj F16 · 0.8 GB"]
            runtime["rizzo/runtimes<br/>llama.cpp b11081 · Metal<br/>libllama · llama-server"]
        end
        p1(["jevnav process<br/>simulation + navigation"])
        p2(["rizzo serve process<br/>FastAPI · port 8017"])
        p3(["llama-server child process<br/>vision mode only · free local port"])
        gpu[("M5 GPU · Metal")]
    end
    results[("results/ — reports and traces")]

    p1 -->|"HTTP JSON · from a background thread, one request in flight"| p2
    p2 -->|"Spark: in-process libllama"| gpu
    p2 -->|"Qwen3-VL: starts it, HTTP /completion with the image"| p3
    p3 --> gpu
    p1 -.->|"runs in"| venv1
    p2 -.->|"runs in"| venv2
    p2 -.->|"loads at start, about 10 s"| weights
    p2 -.->|"loads"| runtime
    p3 -.->|"loads Qwen3-VL + mmproj"| weights
    p1 -.->|"imports"| src
    p1 -.->|"imports"| ext
    p1 -->|"--report / --trace / --output"| results
```

## 3 · Components inside JEVNAV

```mermaid
flowchart TB
    cli["CLI<br/>run · bench · check · scenarios"]
    viz["Visualizer<br/>ir-sim view + decision sidebar · GIF / PNG<br/>operator instruction box"]
    nav["Navigator<br/>episode lifecycle · adaptive hold H = 1.2 × latency"]

    subgraph route["Global route — see features/global-route.md"]
        occ["Occupancy<br/>static map → clearance field"]
        plan["RoutePlanner<br/>clearance-aware A* + shortcut"]
    end

    subgraph local["Local costmap — see features/local-costmap.md"]
        build["CostmapBuilder<br/>ray-cast latest scan · inflate · overlay route and goal"]
        enc["CostmapEncoder<br/>grid: one text map + legend<br/>text: coordinates in the robot frame"]
        img["CostmapImage<br/>image: 1344 × 1344 PNG in the viewer's colors"]
    end

    subgraph simulation["Simulation decision — see features/simulation-decision.md"]
        sim["CommandSimulator<br/>24 commands over H → end poses"]
        safe["SafetyFilter<br/>collision · clearance · kinematics<br/>before Rizzo and before execution"]
        senc["SimulationEncoder<br/>route to go + goal distance per safe command"]
    end

    cmds["CommandSet<br/>24 fixed commands — see features/command-set.md"]

    subgraph decide["Decision — see features/rizzo-decision.md"]
        worker["DecisionWorker<br/>background thread · runs while the robot moves"]
        dec{{"Decider interface"}}
        jd["JevDecider<br/>HTTP client for /v1/systemone"]
        hd["HeuristicDecider<br/>classical baseline"]
    end

    rep["EpisodeReport + DecisionTrace"]
    irsim[["ir-sim env"]]
    rizzo[["Rizzo Flow server"]]

    cli -->|"scenario, seed, decider"| nav
    nav -.->|"observer: request, decision, step"| viz
    viz -->|"operator instruction, see features/operator-instruction.md"| nav
    viz -->|"render"| irsim
    irsim -->|"pose, scan, arrive / collision flags"| nav
    nav -->|"static obstacles, grid map"| occ
    occ -->|"clearance field"| plan
    plan -->|"Route"| build
    nav -->|"latest scan + predicted pose, at each command switch"| build
    build -->|"LocalCostmap"| enc
    enc -->|"evidence text"| worker
    build -->|"LocalCostmap"| img
    img -->|"image next to the simulation state, see features/image-evidence.md"| worker
    cmds -->|"24 commands"| sim
    nav -->|"latest scan + predicted pose"| sim
    sim --> safe
    plan -->|"Route"| senc
    safe -->|"safe commands"| senc
    senc -->|"state + one choice question"| worker
    worker --> dec
    build -->|"LocalCostmap"| hd
    cmds -->|"24 commands"| dec
    dec --- jd
    dec --- hd
    jd <-->|"POST /v1/systemone"| rizzo
    worker -->|"Decision: ranked probabilities + measured latency"| nav
    nav -->|"at the hold end: ranking, actual pose, fresh scan<br/>simulation · image evidence"| safe
    safe -->|"first safe command, or safety stop"| nav
    nav ==>|"executed command v, ω<br/>grid · text: most probable as is<br/>env.step every 0.1 s for H"| irsim
    nav -->|"outcome, metrics, trajectory, every decision"| rep
```

## 4 · Key data

```mermaid
classDiagram
    class Scan {
        lidar_origin_world
        ranges_180
        angles
        range_max
    }
    class LocalCostmap {
        predicted_pose
        layers_21x21
        route_mask
        goal_cell
        route_points
        goal_point
    }
    class Route {
        points
        stations_arc_length
        project(xy)
        point_at(station)
    }
    class Command {
        id
        direction
        linear_m_s
        angular_rad_s
        text()
    }
    class Verdict {
        safe
        reason
        clearance
    }
    class DecisionRequest {
        index
        sent_step
        evidence
        instructions
        options_shuffled
        hold
        evidence_format
        descriptions
        instruction
        images
    }
    class Decision {
        command_id
        probabilities
        ranking
        confidence
        latency_s
        input_tokens
        served_by
    }
    class DecisionRecord {
        sent_at_s
        applied_at_s
        executed
        fallbacks
        hold_s
        overrun_s
    }
    class EpisodeReport {
        outcome
        hold_s
        overruns
        safety_stops
        metrics
        trajectory
    }
    LocalCostmap ..> Scan : built from
    LocalCostmap ..> Route : overlay
    DecisionRequest "1" o-- "1" LocalCostmap
    DecisionRequest "1" o-- "0..24" Command : options, all 24 for grid · text
    DecisionRequest "1" o-- "24" Verdict : simulation · image only
    Decision ..> Command : ranks
    DecisionRecord ..> DecisionRequest : records
    DecisionRecord ..> Decision : records
    EpisodeReport "1" o-- "*" DecisionRecord : trace
```

## Notes

- **Robot:** differential drive only (command = linear v, angular ω). Linear speed is fixed
  at 0.25 m/s; rotation in place is 45°/s (π/4 rad/s). The bundled scenarios use a 0.2 m
  radius and a 360° lidar with 180 beams and 5 m range.
- **Rizzo sees one snapshot, never a history.** With `grid` evidence it reads the latest
  costmap only, with the route and the goal drawn into it (see features/local-costmap.md);
  `text` describes the same costmap as coordinates in the robot frame (see
  features/text-evidence.md). `simulation` replaces the costmap with each safe command's
  simulated results, and `image` adds the costmap to them as a picture. Moving obstacles look
  static to the model in every mode.
- **No safety net with grid and text evidence.** The most probable command is executed as is,
  and a collision ends the episode. This is an explicit experiment choice. Simulation and image
  evidence revalidate the ranking from the actual pose at the hold end and fall back to the next
  safe command, or a safety stop (features/simulation-decision.md).
- **Known risk:** Rizzo Flow's README reports that the 4B model plays Snake well with
  per-move sensor text, but dies within 13–26 moves with an ASCII grid only. This design tests
  grid-only input on navigation. The HeuristicDecider runs on the same costmap and the
  same 24 commands, so `bench` measures how much the model gains or loses against it.
- **Decisions arrive while the robot moves.** Each command is held for H = 1.2 × Rizzo's
  latency, adapted at every decision (1.6–1.9 s on the M5, where a request takes 1.3–1.6 s
  and slows under sustained load). The next decision is
  computed during the hold from a costmap drawn around the predicted pose, and applied when
  the hold ends. Each request's measured latency is replayed in simulated time, so the robot
  behaves as in real time whatever the simulator's own speed (see features/rizzo-decision.md).
  For slow models, `--sim-latency S` counts every answer as S seconds instead and pauses the
  simulation while the model thinks (fake timing).
- **Snake logic** (`--evidence simulation`, and `image` on top of it). As in Rizzo's Snake
  demo, the simulator computes what every command would do, and Rizzo only chooses. Unsafe
  commands (collision, clearance under 0.05 m, velocity limits) are removed first; each safe
  one is offered with the route still to go, the straight-line goal distance and the closest
  obstacle after it (see features/simulation-decision.md).
- **Jev wire format.** JEVNAV only uses `/v1/systemone`, so the hosted Jev works by changing
  `--jev-url`. 24 options fit Rizzo's limit of 26 answer letters (A–X are used).
  `--evidence image` adds an `images` field that only the local Rizzo's vision mode accepts.
- **Vision mode** (Rizzo Flow branch `vlm-costmap-image`). `rizzo serve --vision` runs Qwen3-VL-4B-Instruct
  (`--model` + `--mmproj` for other files) through the runtime's own llama-server, which
  handles image tokens; Rizzo keeps its prompt, answer letters and softmax
  (features/image-evidence.md).
- **Visualizer** (`jevnav run --render`): ir-sim's own view on the left (robot, lidar beams,
  obstacles, trajectory) plus the global route, the command being held and the pose where the
  next decision starts. The sidebar shows the status (including "paused, waiting for the
  model" while an answer is late), a 24-direction probability compass, the exact costmap of the
  latest decision with the held command's path (titled for the evidence mode), and the decision
  log. `--speed` sets the real-time playback factor; `--save-gif` and `--save-frame` work
  headless.
  An instruction box and preset buttons at the top left, above the simulation, add an operator
  instruction to every new request's question (features/operator-instruction.md).
  The window comes to the front once, when the run starts. Every later frame leaves it where
  it is (matplotlib's `figure.raise_window` is turned off; otherwise each `plt.pause` raises it
  on macOS), and after the run it stays open until closed without being raised again
  (`plt.show()` would activate the app), so it can sit behind other windows and never takes
  the focus back.
