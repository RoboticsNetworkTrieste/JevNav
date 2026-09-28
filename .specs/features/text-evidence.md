# Text evidence (experiment 1)

## Goal

Describe the same local costmap to Rizzo in words and coordinates instead of an ASCII grid:
the robot's pose and heading, the goal, the planned path and the obstacle points in the
window, all relative to the robot. It tests whether the 4B model, which does not read a grid
spatially, decides better from coordinates. Selected with `--evidence text`; the grid stays
the default.

## Flow

```mermaid
flowchart TD
    CM["LocalCostmap<br/>21 × 21 · 0.2 m · centred on the predicted pose<br/>(features/local-costmap.md)"] --> SEL{"--evidence"}
    SEL -->|"grid (default)"| GRID["Header + legend + 21 text rows"]
    SEL -->|"text"| FR["Robot frame at the predicted pose<br/>x ahead · y left · meters"]
    FR --> RB["Robot line<br/>world position and heading, radius"]
    FR --> GO["Goal line<br/>goal in the robot frame, even outside the window · distance"]
    FR --> PA["Path line<br/>visible route resampled every 0.4 m, in order toward the goal"]
    FR --> OB["Obstacle line<br/>centre of every # cell in the window · nearest first"]
    RB --> TXT["Evidence text · one line per item"]
    GO --> TXT
    PA --> TXT
    OB --> TXT
    GRID --> ST["state field of the Rizzo request"]
    TXT --> ST
    SEL -->|"text"| IN["Instructions in words:<br/>follow the path · closer than the radius is a collision ·<br/>keep 0.6 m when you can"]
    IN --> Q["instructions field of the Rizzo request"]
```

## Robot frame

```
            +x  (ahead: F)
             ▲
             │   (1.2, 0.4)
   +y ◄──────R      y > 0: left   (L, FL…)
  (left)     │      y < 0: right  (R, FR…)
             │      x < 0: behind (B…)
```

The frame is the robot at the predicted pose, the same pose the grid is centred on. The
command texts use the same words (forward, left, right, backward).

## Evidence text (the `state` field)

```
A differential-drive robot of radius 0.2 m, described where it will be when the chosen command starts.
Robot: at (3.2, 1.5) in the world, heading 45 degrees.
Coordinates below are in meters relative to the robot: x is ahead (negative: behind), y is to the left (negative: right).
Goal: (4.1, 2.3), 4.7 m away.
Planned path to the goal, a point every 0.4 m, nearest first: (0.0, 0.1) (0.4, 0.2) (0.8, 0.4) (1.1, 0.7) (1.4, 1.0) (1.6, 1.3)
Obstacles seen by the lidar within 2 m ahead, behind or to either side, nearest first: (0.6, -0.4) (0.6, -0.2) (0.8, -0.4) …
```

- Every coordinate is rounded to 0.1 m. Obstacle points are cell centres, so they sit on a
  0.2 m lattice.
- Empty lines say `none`: `Planned path …: none in view` when the robot is over 2 m off the
  route, `Obstacles …: none` when the lidar sees nothing in the window.

## Instructions (text mode)

```
Choose the motion command the robot will hold for the next 1.7 s. Every move is at 0.25 m/s
and rotations turn 45 degrees per second. Follow the planned path toward the goal. An obstacle
point closer than 0.2 m to the robot centre is a collision; keep at least 0.6 m from every
obstacle point when you can.
```

## Notes

- **Same information, minus unknown space.** The obstacle points are the grid's `#` cells,
  the path is the grid's `*` cells resampled every 0.4 m, and 0.2 m and 0.6 m are the
  grid's `x` and `+` thresholds. What is dropped is the free-versus-unknown distinction
  (`.` and `?`). What is added is the goal position when the goal is outside the window.
- **Token cost grows with the obstacles** (measured 2026-09-24, 12 maps from the three
  scenarios, warm server): about 9 tokens per obstacle point, instead of a fixed 141 tokens
  for the grid. Scenarios show 0–43 points per map, 23 at the median in `open` and `slalom`.

  | Evidence | Input tokens p50 | Latency p50 | 35–41 obstacle points |
  | --- | ---: | ---: | --- |
  | grid | 995 | 1.25 s | 1,014–1,022 tokens, 1.28 s |
  | text | 1,082 | 1.40 s | 1,282–1,292 tokens, 1.71–1.72 s |

  Crowded maps therefore get longer holds (up to about 2.1 s instead of 1.5 s).
- The HeuristicDecider ignores the evidence text (it reads the LocalCostmap), so its results
  are the same with `grid` and `text`. With `simulation` or `image` it chooses among the safe
  commands only, and its choice is revalidated.
- Reports and `bench` tables record the evidence mode, so grid and text runs can be compared.

## Results (2026-09-24, seeds 0–2, `runs/exp1-text.json` and `runs/exp1-grid.json`)

Run back to back with a fresh grid benchmark in the same thermal state. Progress is the
closest approach to the goal as a share of the start distance.

| Scenario | Rizzo · text | Rizzo · grid (same session) |
| --- | --- | --- |
| open | 0/3, mean progress 10% | 0/3, 21% |
| slalom | 0/3, 21% | 0/3, 14% |
| crossing | 0/3, 10% | 0/3, 22% |

- No difference beyond noise. The same seed doesn't reproduce: text `slalom` seed 1 went
  18.8 m in the benchmark and 7.0 m in a rerun, because a 0.1 s shift in one command switch
  changes the rest of the run.
- Chosen commands: text F 42%, backward 32%; grid F 34–38%, backward 35–37%. The heuristic
  chose backward once in 204 decisions.
- The same map can get a clearly wrong answer: with the path straight ahead and every
  obstacle within 1 m behind, Rizzo chose BL45 (p 0.46) twice in a row.

## Open Questions

- none
