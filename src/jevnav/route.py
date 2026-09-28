from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Route:
    points: np.ndarray
    stations: np.ndarray

    @classmethod
    def through(cls, points) -> "Route":
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return cls(points, np.concatenate([[0.0], np.cumsum(segment_lengths)]))

    @property
    def length(self) -> float:
        return float(self.stations[-1])

    @property
    def goal(self) -> np.ndarray:
        return self.points[-1]

    def project(self, xy, lower: float = -np.inf, upper: float = np.inf) -> tuple[float, float]:
        xy = np.asarray(xy, dtype=float)
        if len(self.points) == 1:
            return 0.0, float(np.linalg.norm(xy - self.points[0]))
        starts = self.points[:-1]
        directions = self.points[1:] - starts
        squared = np.maximum(np.einsum("ij,ij->i", directions, directions), 1e-12)
        fractions = np.clip(np.einsum("ij,ij->i", xy - starts, directions) / squared, 0.0, 1.0)
        closest = starts + fractions[:, None] * directions
        distances = np.linalg.norm(xy - closest, axis=1)
        stations = self.stations[:-1] + fractions * (self.stations[1:] - self.stations[:-1])
        in_window = (self.stations[1:] >= lower) & (self.stations[:-1] <= upper)
        if not in_window.any():
            in_window[:] = True
        best = int(np.argmin(np.where(in_window, distances, np.inf)))
        return float(stations[best]), float(distances[best])

    def point_at(self, station: float) -> np.ndarray:
        station = float(np.clip(station, 0.0, self.length))
        return np.array(
            [
                np.interp(station, self.stations, self.points[:, 0]),
                np.interp(station, self.stations, self.points[:, 1]),
            ]
        )

    def heading_at(self, station: float) -> float:
        if len(self.points) == 1:
            return 0.0
        segment = int(
            np.clip(
                np.searchsorted(self.stations, station, side="right") - 1, 0, len(self.points) - 2
            )
        )
        direction = self.points[segment + 1] - self.points[segment]
        return float(np.arctan2(direction[1], direction[0]))
