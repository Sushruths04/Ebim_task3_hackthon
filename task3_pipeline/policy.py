# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""The retry policy: choose the next parameter set to try after a failure.

This is the automated version of a human reading the changelog and hand-editing
one constant. It is FAR / Reflexion "retry-with-adjustment": pick the next
candidate from a small bounded grid, best-known-first (from memory), skipping
combinations already known to fail, and biasing the choice by the *diagnosis*
of the last failure (e.g. an IK_FAIL flips the stance before touching offsets).

Pure logic, fully CPU-testable. No training, no GPU.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from task3_pipeline.config import GRIDS, RETRY_BUDGET, ParamGrid
from task3_pipeline.memory import ParamMemory
from task3_pipeline.outcomes import SkillOutcome, SkillReport


def _grid_candidates(grid: ParamGrid) -> list[dict]:
    """Cartesian product of a grid, in declaration order (best-first)."""
    if not grid.grid:
        return [{}]
    keys = list(grid.grid.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*grid.grid.values())
    ]


# Diagnosis-driven reordering: when the last failure points at a cause, bring
# the knob that fixes it to the front. This is the "innovative" bit -- the loop
# reasons about *why* it failed instead of blindly sweeping.
_OUTCOME_PRIORITY_KEY = {
    SkillOutcome.IK_FAIL: "approach_stance",  # unreachable -> change stance
    SkillOutcome.WEAK_GRASP: "grasp_height_above_origin_m",
    SkillOutcome.MISS: "grasp_y_offset",
    SkillOutcome.SLIP: "base_hold_kp",
    SkillOutcome.NAV_SHORT: "max_linear_mps",
}


@dataclass
class RetryPolicy:
    memory: ParamMemory
    budget: int = RETRY_BUDGET

    def plan(
        self,
        skill: str,
        *,
        head_placement: str = "-",
        object_name: str = "-",
        last: SkillReport | None = None,
    ) -> list[dict]:
        """Return an ordered list of parameter sets to try (len <= budget+1).

        Order: best-known-from-memory -> diagnosis-prioritised grid ->
        remaining grid, with known-failing combos pushed to the back.
        """
        grid = GRIDS.get(skill, ParamGrid(name=skill))
        candidates = _grid_candidates(grid)

        # Reorder grid by the knob that addresses the last failure mode.
        if last is not None:
            priority_key = _OUTCOME_PRIORITY_KEY.get(last.outcome)
            if priority_key and priority_key in grid.grid:
                candidates.sort(
                    key=lambda c, k=priority_key, lp=last.params: (
                        c.get(k) == lp.get(k)  # try a DIFFERENT value first
                    )
                )

        # De-prioritise parameter sets already known to fail here.
        failed = self.memory.failed_params(
            skill, head_placement=head_placement, object_name=object_name
        )
        candidates.sort(key=lambda c: c in failed)

        # Seed with the best-known-good params for this exact context.
        ordered: list[dict] = []
        best = self.memory.best_params(
            skill, head_placement=head_placement, object_name=object_name
        )
        if best is not None:
            ordered.append(best)
        for c in candidates:
            if c not in ordered:
                ordered.append(c)

        return ordered[: self.budget + 1]
