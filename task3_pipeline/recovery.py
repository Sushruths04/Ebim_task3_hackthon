# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed failure -> recovery action table (REV20 P4.5, P5).

WHY THIS IS A NEW LAYER AND NOT A NEW ENTRY IN `policy.py`'s TABLE.
`policy._OUTCOME_PRIORITY_KEY` answers one question well: *which knob do I turn
before retrying?* That presumes the answer is always "retry with a different
parameter". For every outcome it currently lists (IK_FAIL, WEAK_GRASP, MISS,
SLIP, NAV_SHORT) that presumption holds.

It does **not** hold for the two perception outcomes REV20 P4.5 adds. When the
pose is stale or untrusted, the action did not go wrong -- it should not have
run at all. Retrying with a different grasp offset would aim a differently
tuned motion at the same bad belief, and would burn retry budget doing it.
REV20 P4.5 states the requirement directly: the planner must trigger
**re-perception instead of acting**.

So this module answers the prior question -- *should I retry at all, or do
something else first?* -- and defers to `RetryPolicy` only for the outcomes
where retrying is genuinely the right move. Folding these into the existing
dict would have silently mapped "I don't know where the object is" onto "nudge
the gripper 2 cm", which is the kind of category error this project's failure
log is already full of.

Pure logic, no Isaac/ROS/torch, fully CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from task3_pipeline.outcomes import SkillOutcome
from task3_pipeline.tracked_object import TrackedObject


class RecoveryAction(str, Enum):
    """What to do next, before any parameter choice is considered."""

    PROCEED = "proceed"  # nothing went wrong
    RETRY_WITH_PARAM = "retry_with_param"  # hand off to RetryPolicy
    REPERCEIVE = "reperceive"  # refresh the belief, then re-decide
    REPLAN_NAV = "replan_nav"  # ask Nav2 for a new path/stance
    ABORT_OBJECT = "abort_object"  # give up on this object, keep the episode


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    # True when this decision must NOT consume the per-object retry budget.
    # Re-perceiving is not an attempt at the task; charging it would let a
    # flickering detector exhaust the budget without the robot ever moving.
    free: bool = False


# The table. Anything absent falls through to RETRY_WITH_PARAM, which keeps
# behaviour identical to today for every outcome that already worked.
_TABLE: dict[SkillOutcome, RecoveryDecision] = {
    SkillOutcome.SUCCESS: RecoveryDecision(
        RecoveryAction.PROCEED, "skill succeeded"
    ),
    SkillOutcome.PERCEPTION_STALE: RecoveryDecision(
        RecoveryAction.REPERCEIVE,
        "pose older than the freshness budget; acting on it risks aiming at "
        "where the object used to be",
        free=True,
    ),
    SkillOutcome.PERCEPTION_LOW_CONFIDENCE: RecoveryDecision(
        RecoveryAction.REPERCEIVE,
        "detection confidence below threshold; refine the belief before "
        "committing motion to it",
        free=True,
    ),
    # Nav is Nav2's job as of P2: a short stop is a planning/control problem,
    # not a grasp-parameter problem, so it goes back to the navigator rather
    # than into the grasp grid.
    SkillOutcome.NAV_SHORT: RecoveryDecision(
        RecoveryAction.REPLAN_NAV, "base stopped outside tolerance"
    ),
    # A timeout has already spent the budget it was going to spend. Retrying
    # the same skill with a new parameter usually just times out again and
    # costs another full budget, which is how episodes reach the 1800 s wall
    # with no EPISODE_RESULT -- a pattern this project has recorded repeatedly.
    SkillOutcome.TIMEOUT: RecoveryDecision(
        RecoveryAction.ABORT_OBJECT,
        "skill exhausted its step budget; move to the next object rather than "
        "spending another full budget on the same one",
    ),
}


def decide(
    outcome: SkillOutcome,
    *,
    reperceive_attempts: int = 0,
    max_reperceive: int = 2,
) -> RecoveryDecision:
    """Choose the recovery action for `outcome`.

    `reperceive_attempts` is how many times re-perception has already been
    tried for this object without producing an actionable belief. Bounding it
    matters: a genuinely occluded or absent object would otherwise loop
    REPERCEIVE forever, which looks like a hang rather than a failure and is
    indistinguishable from one in the logs. Past the bound the object is
    abandoned so the episode can still score on the others.
    """
    if max_reperceive < 0:
        raise ValueError(f"max_reperceive must be >= 0, got {max_reperceive}")

    decision = _TABLE.get(
        outcome,
        RecoveryDecision(
            RecoveryAction.RETRY_WITH_PARAM,
            f"{outcome.value}: retry with an adjusted parameter",
        ),
    )

    if (
        decision.action is RecoveryAction.REPERCEIVE
        and reperceive_attempts >= max_reperceive
    ):
        return RecoveryDecision(
            RecoveryAction.ABORT_OBJECT,
            f"re-perception did not yield an actionable pose after "
            f"{reperceive_attempts} attempts ({outcome.value}); abandoning "
            "this object so the episode can continue",
        )
    return decision


def gate_from_track(
    track: TrackedObject | None,
    now_s: float,
    *,
    max_age_s: float,
    tau: float,
) -> SkillOutcome | None:
    """Build a `SelfCorrectingSkill(perception_gate=...)` result from a track.

    This is the seam between the belief (`TrackedObject`) and the decision
    (`decide`). It exists as a named function rather than a lambda at each call
    site so there is exactly one place where "is this belief good enough to act
    on" is answered -- the alternative is every skill re-implementing the gate
    slightly differently, which is how thresholds drift apart unnoticed.

    A missing track is treated as low confidence rather than an error: an
    object the detector has not seen yet is exactly the case re-perception is
    for, and raising here would abort the episode instead of looking again.
    """
    if track is None:
        return SkillOutcome.PERCEPTION_LOW_CONFIDENCE
    return perception_outcome(
        is_stale=track.is_stale(now_s, max_age_s),
        is_confident=track.is_confident(tau),
    )


def perception_outcome(
    *, is_stale: bool, is_confident: bool
) -> SkillOutcome | None:
    """Map a `TrackedObject` gate result onto an outcome, or None if fine.

    Staleness is reported ahead of low confidence when both hold, because it is
    the more actionable of the two: a fresh look fixes staleness outright,
    whereas low confidence may persist and eventually justify abandoning the
    object.
    """
    if is_stale:
        return SkillOutcome.PERCEPTION_STALE
    if not is_confident:
        return SkillOutcome.PERCEPTION_LOW_CONFIDENCE
    return None
