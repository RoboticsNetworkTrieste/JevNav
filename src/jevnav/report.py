from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DecisionRecord:
    index: int
    sent_at: float
    applied_at: float | None
    command_id: str
    probabilities: dict[str, float]
    latency: float
    confidence: float | None
    input_tokens: int | None
    overrun: float
    option_order: list[str]
    evidence: str
    predicted_pose: list[float]
    hold: float | None = None
    instructions: str = ""
    executed: str | None = None
    fallbacks: int = 0
    ranking: list[str] = field(default_factory=list)
    removed: dict[str, str] = field(default_factory=dict)
    option_texts: dict[str, str] = field(default_factory=dict)
    instruction: str | None = None
    image_png: str | None = None

    @property
    def probability(self) -> float:
        return self.probabilities.get(self.command_id, 0.0)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "sent_at_s": round(self.sent_at, 2),
            "applied_at_s": None if self.applied_at is None else round(self.applied_at, 2),
            "choice": self.command_id,
            "probability": round(self.probability, 4),
            "probabilities": {key: round(value, 4) for key, value in self.probabilities.items()},
            "confidence": self.confidence,
            "latency_s": round(self.latency, 4),
            "input_tokens": self.input_tokens,
            "hold_s": self.hold,
            "overrun_s": round(self.overrun, 2),
            "option_order": self.option_order,
            "predicted_pose": self.predicted_pose,
            "executed": self.executed,
            "fallbacks": self.fallbacks,
            "ranking": self.ranking,
            "removed": self.removed,
            "option_texts": self.option_texts,
            "operator_instruction": self.instruction,
            "evidence": self.evidence,
            "instructions": self.instructions,
            "image_png": self.image_png,
        }


@dataclass
class EpisodeReport:
    scenario: str
    decider: str
    seed: int | None = None
    evidence_format: str = "grid"
    sim_latency: float | None = None
    served_by: str | None = None
    outcome: str = "running"
    hold: float | None = None
    warmup_latencies: list[float] = field(default_factory=list)
    time_limit: float | None = None
    sim_time: float = 0.0
    distance_traveled: float = 0.0
    route_length: float | None = None
    straight_line: float = 0.0
    min_clearance: float = float("inf")
    replans: int = 0
    decisions: list[DecisionRecord] = field(default_factory=list)
    trajectory: list[list[float]] = field(default_factory=list)

    @property
    def applied(self) -> list[DecisionRecord]:
        return [record for record in self.decisions if record.applied_at is not None]

    @property
    def overruns(self) -> int:
        return sum(record.overrun > 0 for record in self.applied)

    @property
    def safety_stops(self) -> int:
        return sum(record.executed == "STOP" for record in self.applied)

    @property
    def fallbacks(self) -> int:
        return sum(record.fallbacks for record in self.applied if record.executed != "STOP")

    @property
    def latencies(self) -> list[float]:
        return [record.latency for record in self.decisions]

    @property
    def hold_median(self) -> float | None:
        holds = [record.hold for record in self.applied if record.hold]
        return float(np.median(holds)) if holds else self.hold

    @property
    def efficiency(self) -> float | None:
        if not self.route_length or self.distance_traveled <= 0 or self.outcome != "arrived":
            return None
        return self.route_length / self.distance_traveled

    def latency_ms(self, quantile: float) -> float | None:
        if not self.latencies:
            return None
        return float(np.quantile(self.latencies, quantile) * 1000)

    def choices(self) -> dict[str, int]:
        return dict(
            Counter(record.executed or record.command_id for record in self.applied).most_common()
        )

    def summary(self) -> dict:
        tokens = [record.input_tokens for record in self.decisions if record.input_tokens]
        return {
            "scenario": self.scenario,
            "decider": self.decider,
            "evidence": self.evidence_format,
            "sim_latency_s": self.sim_latency,
            "served_by": self.served_by,
            "seed": self.seed,
            "outcome": self.outcome,
            "initial_hold_s": self.hold,
            "hold_p50_s": _finite(self.hold_median, 2),
            "warmup_latencies_s": [round(value, 3) for value in self.warmup_latencies],
            "time_limit_s": None if self.time_limit is None else round(self.time_limit, 1),
            "sim_time_s": round(self.sim_time, 2),
            "distance_traveled_m": round(self.distance_traveled, 2),
            "route_length_m": None if self.route_length is None else round(self.route_length, 2),
            "straight_line_m": round(self.straight_line, 2),
            "path_efficiency": None if self.efficiency is None else round(self.efficiency, 3),
            "min_clearance_m": _finite(self.min_clearance, 3),
            "decisions": len(self.applied),
            "overruns": self.overruns,
            "safety_stops": self.safety_stops,
            "validation_fallbacks": self.fallbacks,
            "replans": self.replans,
            "latency_p50_ms": _finite(self.latency_ms(0.5), 1),
            "latency_p95_ms": _finite(self.latency_ms(0.95), 1),
            "input_tokens_p50": int(np.median(tokens)) if tokens else None,
            "choices": self.choices(),
        }

    def to_dict(self) -> dict:
        data = self.summary()
        data["trajectory"] = self.trajectory
        return data

    def trace(self) -> list[dict]:
        return [record.to_dict() for record in self.decisions]

    def line(self) -> str:
        summary = self.summary()
        efficiency = summary["path_efficiency"]
        latency = summary["latency_p50_ms"]
        clearance = summary["min_clearance_m"]
        return (
            f"{self.scenario:<9} {self.decider:<9} {self.evidence_format:<4} "
            f"seed={self.seed!s:<3} {self.outcome:<11} "
            f"hold={self.hold_median or 0:.1f}s  t={self.sim_time:>6.1f}s  "
            f"dist={self.distance_traveled:>5.1f}m  route={self.route_length or 0:>5.1f}m  "
            f"eff={'-' if efficiency is None else f'{efficiency:.2f}':>4}  "
            f"minclr={'-' if clearance is None else f'{clearance:.2f}m'}  "
            f"decisions={len(self.applied)} p50={'-' if latency is None else f'{latency:.0f}ms'} "
            f"overruns={self.overruns} replans={self.replans} stops={self.safety_stops}"
        )


def _finite(value, digits: int):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def aggregate(reports: list[EpisodeReport]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[EpisodeReport]] = {}
    for report in reports:
        key = (report.scenario, report.decider, report.evidence_format)
        groups.setdefault(key, []).append(report)
    rows = []
    for (scenario, decider, evidence_format), items in groups.items():
        arrived = [item for item in items if item.outcome == "arrived"]
        efficiencies = [item.efficiency for item in arrived if item.efficiency is not None]
        latencies = [value for item in items for value in item.latencies]
        rows.append(
            {
                "scenario": scenario,
                "decider": decider,
                "evidence": evidence_format,
                "episodes": len(items),
                "success_rate": round(len(arrived) / len(items), 3),
                "collisions": sum(item.outcome == "collided" for item in items),
                "timeouts": sum(item.outcome == "timed_out" for item in items),
                "mean_time_s": _mean([item.sim_time for item in arrived]),
                "mean_efficiency": _mean(efficiencies, 3),
                "latency_p50_ms": _finite(float(np.quantile(latencies, 0.5) * 1000), 1)
                if latencies
                else None,
                "overruns": sum(item.overruns for item in items),
                "safety_stops": sum(item.safety_stops for item in items),
            }
        )
    return rows


def _mean(values, digits: int = 2):
    return round(float(np.mean(values)), digits) if values else None
