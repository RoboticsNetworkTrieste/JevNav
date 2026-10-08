import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .costmap import NEAR, OBSTACLE, TOO_CLOSE, UNKNOWN
from .kinematics import rollout
from .prompt import DecisionRequest
from .route import Route
from .safety import SAFETY_STOP

QUESTION_ID = "command"
DEFAULT_JEV_URL = "http://127.0.0.1:8700"
DEFAULT_JEV_MODEL = "clm-latest"
SAFETY = "safety"


@dataclass(frozen=True)
class Decision:
    command_id: str
    probabilities: dict[str, float]
    latency: float
    confidence: float | None = None
    input_tokens: int | None = None
    served_by: str | None = None
    ranking: tuple[str, ...] = ()

    def ranked(self) -> tuple[str, ...]:
        if self.ranking:
            return self.ranking
        others = sorted(
            (key for key in self.probabilities if key != self.command_id),
            key=lambda key: -self.probabilities[key],
        )
        return (self.command_id, *others)


def safety_stop() -> Decision:
    return Decision(SAFETY_STOP.id, {}, 0.0, served_by=SAFETY)


class Decider(Protocol):
    name: str

    def decide(self, request: DecisionRequest) -> Decision: ...


class JevUnavailable(RuntimeError):
    pass


class JevDecider:
    name = "jev"

    def __init__(
        self,
        base_url: str = DEFAULT_JEV_URL,
        model: str = DEFAULT_JEV_MODEL,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def served_models(self) -> list[str]:
        payload = self._call("GET", "/v1/models")
        entries = payload.get("models") or payload.get("data") or []
        return [entry.get("name") or entry.get("id", "") for entry in entries]

    def request_body(self, request: DecisionRequest) -> dict:
        images = {"images": list(request.images)} if request.images else {}
        return {
            "state": request.evidence,
            **images,
            "model": self.model,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": request.instructions,
                    "criteria": request.criteria(),
                }
            },
        }

    def decide(self, request: DecisionRequest) -> Decision:
        started = time.perf_counter()
        payload = self._call("POST", "/v1/systemone", self.request_body(request))
        latency = time.perf_counter() - started
        answer = payload["answers"][QUESTION_ID]
        choice = answer.get("choice")
        if choice not in request.criteria():
            raise JevUnavailable(
                f"Decider answered {choice!r}, not one of the {len(request.options)} commands"
            )
        probabilities = {
            key: float(value) for key, value in answer.get("probabilities", {}).items()
        }
        return Decision(
            command_id=choice,
            probabilities=probabilities,
            latency=latency,
            confidence=answer.get("confidence"),
            input_tokens=(payload.get("usage") or {}).get("input_tokens"),
            served_by=payload.get("model"),
            ranking=tuple(sorted(probabilities, key=lambda key: -probabilities[key])),
        )

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        call = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(call, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise JevUnavailable(f"{method} {path} → HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise JevUnavailable(f"No Jev-compatible server at {self.base_url}: {error}") from error


class HeuristicDecider:
    name = "heuristic"

    def __init__(
        self,
        step: float = 0.1,
        offset_penalty: float = 0.5,
        near_penalty: float = 0.3,
        unknown_penalty: float = 0.2,
        temperature: float = 0.1,
        latency: float | None = None,
    ):
        self.step = step
        self.latency = latency
        self.offset_penalty = offset_penalty
        self.near_penalty = near_penalty
        self.unknown_penalty = unknown_penalty
        self.temperature = temperature

    def score(self, request: DecisionRequest, command) -> float:
        costmap = request.costmap
        path = rollout(costmap.pose, command.linear, command.angular, request.hold, self.step)
        layers = costmap.layer_at(path[:, :2])
        blocked = np.isin(layers, (OBSTACLE, TOO_CLOSE))
        if blocked[1:].any():
            return -1e3 - float(np.count_nonzero(blocked))
        progress, offset = self._route_progress(costmap.route_points, path)
        near = float(np.mean(layers == NEAR))
        unknown = float(np.mean(layers == UNKNOWN))
        return (
            progress
            - self.offset_penalty * offset
            - self.near_penalty * near
            - self.unknown_penalty * unknown
        )

    def _route_progress(self, route_points: np.ndarray, path: np.ndarray) -> tuple[float, float]:
        if len(route_points) < 2:
            return 0.0, 0.0
        visible = Route.through(route_points)
        start, _ = visible.project(path[0, :2])
        end, offset = visible.project(path[-1, :2])
        return end - start, offset

    def decide(self, request: DecisionRequest) -> Decision:
        started = time.perf_counter()
        scores = {command.id: self.score(request, command) for command in request.options}
        top = max(scores.values())
        weights = {key: math.exp((value - top) / self.temperature) for key, value in scores.items()}
        total = sum(weights.values())
        probabilities = {key: weight / total for key, weight in weights.items()}
        ranking = tuple(sorted(probabilities, key=lambda key: -probabilities[key]))
        return Decision(
            command_id=ranking[0],
            probabilities=probabilities,
            latency=self.latency if self.latency is not None else time.perf_counter() - started,
            served_by=self.name,
            ranking=ranking,
        )
