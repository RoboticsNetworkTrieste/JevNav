import contextlib
import io
import warnings
from pathlib import Path

import numpy as np

from .occupancy import OccupancyGrid, rasterize

SCENARIO_DIR = Path(__file__).parent / "scenarios"
MAX_CELLS_PER_AXIS = 320
MIN_RESOLUTION = 0.1


def available_scenarios() -> list[str]:
    return sorted(path.stem for path in SCENARIO_DIR.glob("*.yaml"))


def resolve_scenario(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()
    bundled = SCENARIO_DIR / f"{candidate.stem}.yaml"
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"No scenario {name_or_path!r}; bundled: {', '.join(available_scenarios())}"
    )


def open_env(path: Path, *, display: bool = False, figure: bool = False, seed: int | None = None):
    import irsim

    return irsim.make(
        str(path),
        display=display,
        headless=not (display or figure),
        seed=seed,
        log_level="WARNING",
    )


def planning_resolution(width: float, height: float) -> float:
    return max(MIN_RESOLUTION, round(max(width, height) / MAX_CELLS_PER_AXIS, 2))


def static_geometries(env) -> list:
    return [
        obstacle.geometry
        for obstacle in env.obstacle_list
        if obstacle.static and obstacle.shape != "map"
    ]


def occupancy_from_env(env, resolution: float | None = None) -> OccupancyGrid:
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        outline = env.get_map(1.0)
        resolution = resolution or planning_resolution(outline.width, outline.height)
        env_map = env.get_map(resolution)
    return rasterize(
        np.asarray(env_map.world_offset, dtype=float),
        env_map.width,
        env_map.height,
        resolution,
        static_geometries(env),
        env_map.grid,
    )
