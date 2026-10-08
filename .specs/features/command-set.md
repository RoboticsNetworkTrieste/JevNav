# Command set

## Goal

CLM always chooses among the same 24 motion commands for a differential-drive robot. Each
command is a **direction of travel** around the robot:
- four full commands: forward, backward, and rotate in place left or right;
- partial turns every 15° in between.

Linear speed is fixed at 0.25 m/s.

## Architecture — direction of travel of each command (top view, robot facing up)

```
                          FL15      F     FR15
                  FL30                            FR30

            FL45                                        FR45

       FL60                                                  FR60

    FL75                                                        FR75

                                    ▲
    L                     ↺       robot       ↻                   R


    BL75                                                        BR75

       BL60                                                  BR60

            BL45                                        BR45

                  BL30                            BR30
                          BL15     B      BR15
```

- **F / B**: straight forward / straight backward.
- **L / R** ("full left / full right"): rotate in place at 45°/s with no forward or backward
  motion. This is how the robot turns around: to drive toward something behind it, it
  rotates with L or R, then moves forward.
- **FLk / FRk**: move forward while curving left / right; the nose turns k/2 degrees per second.
- **BLk / BRk**: move backward toward the rear-left / rear-right, k degrees off straight back.
  The rear swings toward the direction of travel, so the nose turns the opposite way.

## Definition

For a travel direction ψ relative to the nose (left positive, 15° steps):

| ψ | v | ω |
| --- | --- | --- |
| 0° | +0.25 m/s | 0 |
| 0° < \|ψ\| < 90° | +0.25 m/s | ω_max · ψ / 90° |
| ±90° | 0 | ±ω_max |
| 90° < \|ψ\| < 180° | −0.25 m/s | −ω_max · sign(ψ) · (180° − \|ψ\|) / 90° |
| 180° | −0.25 m/s | 0 |

ω_max = **45°/s** (π/4 rad/s), lowered from 90°/s on 2026-09-24. Speed and turn rate come from
the robot's `vel_max` in the scenario YAML. Each command is held for its hold time H = 1.2 ×
the model's latency (see features/model-decision.md); with H = 1.8 s, in one hold the
robot drives 0.45 m, and L / R rotate 81°.

| ID | travel direction | v (m/s) | ω (°/s) | ω (rad/s) | nose turn in a 1.8 s hold | turn radius | text sent to CLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F | 0° | +0.25 | 0 | 0 | 0° | straight | Full forward: straight ahead. |
| FL15 | +15° | +0.25 | +7.5 | +0.131 | +13.5° | 1.91 m | Forward, curving left 7.5 degrees per second. |
| FL30 | +30° | +0.25 | +15 | +0.262 | +27° | 0.95 m | Forward, curving left 15 degrees per second. |
| FL45 | +45° | +0.25 | +22.5 | +0.393 | +40.5° | 0.64 m | Forward, curving left 22.5 degrees per second. |
| FL60 | +60° | +0.25 | +30 | +0.524 | +54° | 0.48 m | Forward, curving left 30 degrees per second. |
| FL75 | +75° | +0.25 | +37.5 | +0.654 | +67.5° | 0.38 m | Forward, curving left 37.5 degrees per second. |
| L | +90° | 0 | +45 | +0.785 | +81° | in place | Full left: rotate in place to the left. |
| BL75 | +105° | −0.25 | −37.5 | −0.654 | −67.5° | 0.38 m | Backward toward the rear-left, 75 degrees off straight back. |
| BL60 | +120° | −0.25 | −30 | −0.524 | −54° | 0.48 m | Backward toward the rear-left, 60 degrees off straight back. |
| BL45 | +135° | −0.25 | −22.5 | −0.393 | −40.5° | 0.64 m | Backward toward the rear-left, 45 degrees off straight back. |
| BL30 | +150° | −0.25 | −15 | −0.262 | −27° | 0.95 m | Backward toward the rear-left, 30 degrees off straight back. |
| BL15 | +165° | −0.25 | −7.5 | −0.131 | −13.5° | 1.91 m | Backward toward the rear-left, 15 degrees off straight back. |
| B | 180° | −0.25 | 0 | 0 | 0° | straight | Full backward: straight back. |
| BR15 | −165° | −0.25 | +7.5 | +0.131 | +13.5° | 1.91 m | Backward toward the rear-right, 15 degrees off straight back. |
| BR30 | −150° | −0.25 | +15 | +0.262 | +27° | 0.95 m | Backward toward the rear-right, 30 degrees off straight back. |
| BR45 | −135° | −0.25 | +22.5 | +0.393 | +40.5° | 0.64 m | Backward toward the rear-right, 45 degrees off straight back. |
| BR60 | −120° | −0.25 | +30 | +0.524 | +54° | 0.48 m | Backward toward the rear-right, 60 degrees off straight back. |
| BR75 | −105° | −0.25 | +37.5 | +0.654 | +67.5° | 0.38 m | Backward toward the rear-right, 75 degrees off straight back. |
| R | −90° | 0 | −45 | −0.785 | −81° | in place | Full right: rotate in place to the right. |
| FR75 | −75° | +0.25 | −37.5 | −0.654 | −67.5° | 0.38 m | Forward, curving right 37.5 degrees per second. |
| FR60 | −60° | +0.25 | −30 | −0.524 | −54° | 0.48 m | Forward, curving right 30 degrees per second. |
| FR45 | −45° | +0.25 | −22.5 | −0.393 | −40.5° | 0.64 m | Forward, curving right 22.5 degrees per second. |
| FR30 | −30° | +0.25 | −15 | −0.262 | −27° | 0.95 m | Forward, curving right 15 degrees per second. |
| FR15 | −15° | +0.25 | −7.5 | −0.131 | −13.5° | 1.91 m | Forward, curving right 7.5 degrees per second. |

## Notes

- Generated from `commands.py`; the table and the code must stay identical.
- The speed (0.25 m/s) and the rotation rate (45°/s) are stated once, in the question
  instructions (`grid`, `text`) or in the state (`simulation`, `image`), not in every option. CLM
  embeds each option text on its own and caches it, so with `grid` and `text` the 24 texts
  cost nothing after the first decision (see features/model-decision.md).
- With `grid` and `text` evidence the IDs and texts are identical at every decision; only
  their order is shuffled. `simulation` and `image` offer only the safe commands, each text
  followed by its simulated results (features/simulation-decision.md).
- There is no stop command among the 24. The safety stop (v = 0, ω = 0) of `simulation` and
  `image` is never offered to the model.
- 24 rather than the 22 first requested: with F, L, B and R all included and even spacing,
  the count must be a multiple of 4 (confirmed 2026-09-24).

## Open Questions

- none
