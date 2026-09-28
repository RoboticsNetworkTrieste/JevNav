import math

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rollout(pose, linear: float, angular: float, duration: float, step: float) -> np.ndarray:
    steps = max(0, math.ceil(duration / step - 1e-9))
    x, y, heading = (float(value) for value in np.asarray(pose, dtype=float)[:3])
    headings_before = heading + angular * step * np.arange(steps)
    xs = np.concatenate([[x], x + np.cumsum(linear * step * np.cos(headings_before))])
    ys = np.concatenate([[y], y + np.cumsum(linear * step * np.sin(headings_before))])
    headings = wrap_angle(heading + angular * step * np.arange(steps + 1))
    return np.column_stack([xs, ys, headings])


def predict(pose, linear: float, angular: float, duration: float, step: float) -> np.ndarray:
    return rollout(pose, linear, angular, duration, step)[-1]
