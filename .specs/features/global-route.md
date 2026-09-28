# Global route

## Goal

Before driving, find a route to the goal that is short and keeps a comfortable distance from
the static obstacles. The route is the path-optimization reference and is drawn as `*` into
every local costmap.

## Flow

```mermaid
flowchart TD
    Y["Scenario YAML<br/>static obstacles and optional grid map"] --> O["Rasterize to an occupancy grid<br/>0.1 m per cell, at most 320 cells per axis"]
    O --> C["Clearance field<br/>distance from every cell to the nearest occupied cell"]
    C --> P{"Passable?<br/>clearance above robot radius + half a cell"}
    P -- no --> X["Blocked"]
    P -- yes --> A["A* over 8 neighbours<br/>step cost = length × (1 + 4 × shortfall²)<br/>shortfall = how deep the cell is inside the 0.6 m comfort distance"]
    A --> S["Line-of-sight shortcut<br/>drop waypoints while the straight segment keeps<br/>min(0.6 m, the narrowest gap of the A* cells it skips) − one cell,<br/>never under the 0.1 m safety margin · stop after 12 misses in a row"]
    S --> RT["Route<br/>polyline + arc-length stations"]
    RT --> OV["Drawn as * in every local costmap"]
    RT --> M["Route length = reference for path efficiency"]
    L["Current lidar points, only when replanning"] -.->|"extra obstacles"| O
    GB["Goal cell blocked"] -.->|"snap to the nearest passable cell within 1 m"| A
```

## Notes

- "Optimized" means the shortest route that stays passable. The proximity cost trades a little
  length for distance from obstacles, and the shortcut removes grid zig-zags.
- The static map is known in advance (from the scenario). Moving obstacles are never part of
  the route; they only show up in the lidar costmaps.
- Path efficiency = initial route length / distance actually traveled (arrived episodes only).
- This is a custom planner rather than ir-sim's `AStarPlanner`, which plans for a point-sized
  robot and scans its open set linearly (slow on large maps).
