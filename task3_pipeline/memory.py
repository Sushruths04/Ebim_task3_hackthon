# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Persistent parameter + failure memory.

This is AGENT_STATE.md promoted from human-readable prose to a file that code
can query. Every attempt is recorded as (skill, context) -> list of
(params, outcome, diagnosis). Before retrying, the policy asks this store for
the best-known parameters for the exact situation (skill + head placement +
object), so hard-won knowledge persists across episodes and machines instead
of living in one engineer's head.

Pure-Python, JSON-backed, no external deps. Safe to commit the JSON file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from task3_pipeline.outcomes import SkillOutcome, SkillReport


def context_key(
    skill: str, *, head_placement: str = "-", object_name: str = "-"
) -> str:
    """Stable key for one (skill, situation). Keep it coarse so learning
    generalises across seeds but not across genuinely different geometry."""
    return f"{skill}|{head_placement}|{object_name}"


@dataclass
class Attempt:
    params: dict
    outcome: str
    diagnosis: str = ""
    reward: float = 0.0  # 1.0 success, partial-credit fraction otherwise

    @staticmethod
    def from_report(report: SkillReport, reward: float) -> Attempt:
        return Attempt(
            params=dict(report.params),
            outcome=report.outcome.value,
            diagnosis=report.diagnosis,
            reward=reward,
        )


@dataclass
class ParamMemory:
    """Query/write best-known params per context; persists to JSON."""

    path: str | None = None
    _store: dict[str, list[Attempt]] = field(default_factory=dict)

    # ---- persistence ---------------------------------------------------- #
    @classmethod
    def load(cls, path: str) -> ParamMemory:
        mem = cls(path=path)
        p = Path(path)
        if p.exists():
            raw = json.loads(p.read_text() or "{}")
            mem._store = {k: [Attempt(**a) for a in v] for k, v in raw.items()}
        return mem

    def save(self) -> None:
        if not self.path:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            k: [asdict(a) for a in v] for k, v in self._store.items()
        }
        p.write_text(json.dumps(serialisable, indent=2, sort_keys=True))

    # ---- write ---------------------------------------------------------- #
    def record(self, report: SkillReport, *, reward: float, **ctx) -> None:
        key = context_key(report.skill, **ctx)
        self._store.setdefault(key, []).append(
            Attempt.from_report(report, reward)
        )

    # ---- query ---------------------------------------------------------- #
    def best_params(self, skill: str, **ctx) -> dict | None:
        """Highest-reward params for this context, if any succeeded-ish."""
        attempts = self._store.get(context_key(skill, **ctx), [])
        if not attempts:
            return None
        best = max(attempts, key=lambda a: a.reward)
        return dict(best.params) if best.reward > 0.0 else None

    def failed_params(self, skill: str, **ctx) -> list[dict]:
        """Params that led to a non-success outcome -- the policy avoids
        re-trying these first (the 'don't repeat the same mistake' rule)."""
        attempts = self._store.get(context_key(skill, **ctx), [])
        return [
            dict(a.params)
            for a in attempts
            if a.outcome != SkillOutcome.SUCCESS.value
        ]

    def summary(self) -> dict[str, dict]:
        """Rolling success stats per context, for reporting / the KB view."""
        out: dict[str, dict] = {}
        for key, attempts in self._store.items():
            n = len(attempts)
            ok = sum(
                1 for a in attempts if a.outcome == SkillOutcome.SUCCESS.value
            )
            out[key] = {
                "attempts": n,
                "success_rate": round(ok / n, 3) if n else 0.0,
            }
        return out
