from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt

OCCUPIED_THRESHOLD = 50.0


@dataclass(frozen=True)
class OccupancyGrid:
    origin: np.ndarray
    resolution: float
    occupied: np.ndarray
    clearance: np.ndarray

    @classmethod
    def from_occupied(cls, origin, resolution: float, occupied: np.ndarray) -> "OccupancyGrid":
        occupied = np.asarray(occupied, dtype=bool)
        if occupied.any():
            clearance = distance_transform_edt(~occupied) * resolution
        else:
            clearance = np.full(occupied.shape, np.inf)
        return cls(np.asarray(origin, dtype=float), float(resolution), occupied, clearance)

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    def index_of(self, xy) -> tuple[int, int]:
        i, j = np.floor((np.asarray(xy, dtype=float) - self.origin) / self.resolution).astype(int)
        return int(i), int(j)

    def center_of(self, i: int, j: int) -> np.ndarray:
        return self.origin + (np.array([i, j], dtype=float) + 0.5) * self.resolution

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.shape[0] and 0 <= j < self.shape[1]

    def clearance_at(self, points) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        indices = np.floor((points - self.origin) / self.resolution).astype(int)
        outside = (
            (indices[:, 0] < 0)
            | (indices[:, 0] >= self.shape[0])
            | (indices[:, 1] < 0)
            | (indices[:, 1] >= self.shape[1])
        )
        i = np.clip(indices[:, 0], 0, self.shape[0] - 1)
        j = np.clip(indices[:, 1], 0, self.shape[1] - 1)
        return np.where(outside, 0.0, self.clearance[i, j])

    def with_points(self, points) -> "OccupancyGrid":
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        occupied = self.occupied.copy()
        indices = np.floor((points - self.origin) / self.resolution).astype(int)
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < self.shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < self.shape[1])
        )
        occupied[indices[valid, 0], indices[valid, 1]] = True
        return OccupancyGrid.from_occupied(self.origin, self.resolution, occupied)


def rasterize(
    origin,
    width: float,
    height: float,
    resolution: float,
    geometries=(),
    grid_values: np.ndarray | None = None,
) -> OccupancyGrid:
    origin = np.asarray(origin, dtype=float)
    nx = round(width / resolution)
    ny = round(height / resolution)
    xs = origin[0] + (np.arange(nx) + 0.5) * resolution
    ys = origin[1] + (np.arange(ny) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    occupied = np.zeros((nx, ny), dtype=bool)
    geometries = [geometry for geometry in geometries if geometry is not None]
    if geometries:
        merged = shapely.union_all(geometries)
        shapely.prepare(merged)
        centers = shapely.points(grid_x.ravel(), grid_y.ravel())
        occupied |= shapely.dwithin(merged, centers, resolution * 0.5).reshape(nx, ny)
    if grid_values is not None:
        occupied |= _sample_grid(grid_values, origin, width, height, grid_x, grid_y)
    return OccupancyGrid.from_occupied(origin, resolution, occupied)


def _sample_grid(grid_values, origin, width, height, grid_x, grid_y) -> np.ndarray:
    values = np.asarray(grid_values, dtype=float)
    cell_x = width / values.shape[0]
    cell_y = height / values.shape[1]
    i = np.clip(np.floor((grid_x - origin[0]) / cell_x).astype(int), 0, values.shape[0] - 1)
    j = np.clip(np.floor((grid_y - origin[1]) / cell_y).astype(int), 0, values.shape[1] - 1)
    return values[i, j] > OCCUPIED_THRESHOLD
