# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""The auto-verifier: turn measured state into a labelled skill outcome.

This is the automated replacement for a human staring at a GIF. Every number
used here (gripper angle, position error, object z) is something the existing
skills already measure and log -- we just classify it into a discrete outcome
and a machine-readable diagnosis so the retry policy can act on it.

All functions are pure and CPU-testable (no Isaac, no torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from task3_pipeline import verification
from task3_pipeline.config import THRESHOLDS


class SkillOutcome(str, Enum):
    SUCCESS = "success"
    WEAK_GRASP = "weak_grasp"  # gripper caught body, not a firm cage
    SLIP = "slip"  # object dropped after being held
    IK_FAIL = "ik_fail"  # target unreachable / IK singular
    MISS = "miss"  # end-effector never reached pre-grasp
    NAV_SHORT = "nav_short"  # base stopped outside tolerance
    TIMEOUT = "timeout"  # skill ran out of steps
    UNSCORED = "unscored"  # ran but scorer predicate not met
    # REV20 P4.5. These two are structurally unlike the outcomes above: those
    # mean "the action ran and went wrong", these mean "the action must not run
    # yet". The belief is untrustworthy, so retrying with a different parameter
    # would just aim a differently-tuned motion at the same bad pose. The
    # correct response is to re-perceive. See recovery.py.
    PERCEPTION_STALE = "perception_stale"  # pose too old to act on
    PERCEPTION_LOW_CONFIDENCE = "perception_low_confidence"  # pose not trusted


FAILURE_OUTCOMES = frozenset(
    o for o in SkillOutcome if o is not SkillOutcome.SUCCESS
)


@dataclass(frozen=True)
class SkillReport:
    """Everything the loop needs to decide what to do next.

    ``diagnosis`` is a short structured hint (e.g. "ee +Y of target,
    +0.061 m") mined later by the Planner/memory -- the automated analogue of
    the AGENT_STATE.md changelog line a human writes by hand today.
    """

    skill: str
    outcome: SkillOutcome
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    diagnosis: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome is SkillOutcome.SUCCESS


# --------------------------------------------------------------------------- #
# Classifiers. Each takes the raw measurement dict a skill produced and
# returns (SkillOutcome, diagnosis). Keep them small and explicit.
# --------------------------------------------------------------------------- #


def classify_navigate(metrics: dict) -> tuple[SkillOutcome, str]:
    """A navigate is SUCCESS either on tight terminal error, or -- ACTIVE_
    BRIEF.md sec 3.5/T3 -- on ending with the object still inside the reach
    envelope: raw distance-to-target alone converted several genuinely
    usable stops (pregrasp/descend went on to succeed from them) into a
    retry storm.
    """
    err = float(metrics.get("terminal_error_m", 1.0))
    if err <= THRESHOLDS.nav_tolerance_m:
        return SkillOutcome.SUCCESS, ""
    object_dist = metrics.get("object_dist_m")
    if (
        object_dist is not None
        and float(object_dist) <= THRESHOLDS.nav_reach_envelope_m
    ):
        return SkillOutcome.SUCCESS, (
            f"stopped {err:.3f} m from target but object is "
            f"{float(object_dist):.3f} m away, inside reach envelope"
        )
    return SkillOutcome.NAV_SHORT, f"stopped {err:.3f} m from target"


def classify_reach(metrics: dict) -> tuple[SkillOutcome, str]:
    err = float(metrics.get("position_error_m", 1.0))
    reached = bool(
        metrics.get("strict_reach", err <= THRESHOLDS.reach_tolerance_m)
    )
    if reached and err <= THRESHOLDS.reach_tolerance_m:
        return SkillOutcome.SUCCESS, ""
    dy = float(metrics.get("ee_dy_m", 0.0))
    side = "+Y" if dy > 0 else "-Y"
    return SkillOutcome.IK_FAIL, f"ee {side} of target, err {err:.3f} m"


def classify_grasp(metrics: dict) -> tuple[SkillOutcome, str]:
    """Honest grasp classifier.

    A grasp is SUCCESS only if ALL of: contact was made, the gripper cage
    angle is tight enough, AND the object is *actually held* -- proven either
    by ``object_follows_ee`` being true or by a measured
    ``object_ee_dist_m`` within ``THRESHOLDS.GRASP_HELD_MAX_DIST_M``. This is
    what closes this project's recurring "gripper closed on empty air" bug:
    a cage that LOOKS closed is not proof the object is between the fingers.
    """
    gripper = float(metrics.get("gripper_rad", 1.0))
    contacted = bool(metrics.get("contact", False))
    if not contacted:
        return SkillOutcome.MISS, "no contact at pre-grasp"
    if gripper > THRESHOLDS.grasp_cage_max_rad:
        return (
            SkillOutcome.WEAK_GRASP,
            f"gripper only {gripper:.3f} rad (need <= "
            f"{THRESHOLDS.grasp_cage_max_rad}) -- caught body not rim",
        )

    follows_ee = metrics.get("object_follows_ee")
    dist = metrics.get("object_ee_dist_m")
    held = bool(follows_ee) or (
        dist is not None and float(dist) <= THRESHOLDS.GRASP_HELD_MAX_DIST_M
    )
    if not held:
        dist_str = f"{float(dist):.3f} m" if dist is not None else "unknown"
        return (
            SkillOutcome.WEAK_GRASP,
            f"gripper closed ({gripper:.3f} rad) but object not held "
            f"(dist {dist_str}) -- likely empty",
        )
    return SkillOutcome.SUCCESS, ""


def classify_lift(metrics: dict) -> tuple[SkillOutcome, str]:
    rise = float(metrics.get("object_rise_m", 0.0))
    ik_ok = bool(metrics.get("ik_ok", True))
    if not ik_ok:
        return SkillOutcome.IK_FAIL, "ik failure during lift (fling risk)"
    if rise < THRESHOLDS.min_lift_m:
        return SkillOutcome.MISS, f"object rose only {rise:.3f} m"
    return SkillOutcome.SUCCESS, ""


_HOLD_EVIDENCE_KEYS = (
    "gripper_position_rad",
    "ee_pos_start",
    "ee_pos_end",
    "object_pos_start",
    "object_pos_end",
    "object_rise_m",
)


def classify_hold(metrics: dict) -> tuple[SkillOutcome, str]:
    drop = float(metrics.get("z_drop_m", 0.0))
    held_s = float(metrics.get("held_seconds", 0.0))
    if drop > THRESHOLDS.slip_drop_m:
        return SkillOutcome.SLIP, f"object fell {drop:.3f} m during hold"
    if held_s < float(metrics.get("required_seconds", 0.0)):
        return SkillOutcome.TIMEOUT, f"held {held_s:.1f}s, needed more"

    # REV19 P1.1 / REV20 P5.1 -- the real hold gate. "Did not fall, and time
    # elapsed" also passes for an object sitting on a surface the gripper
    # never closed on, which is exactly the `"ok": true` != a real hold
    # failure this project has recorded repeatedly.
    # `three_predicate_hold` requires gripper-in-cage-band AND
    # object-followed-the-EE AND object-rose, with no partial credit.
    #
    # Applied only when the world supplies the evidence: a world that does not
    # emit these keys keeps today's behaviour rather than being failed for an
    # instrumentation gap. Silently *passing* on missing evidence is the bug
    # being fixed; silently *failing* on it would just be a different one.
    if all(k in metrics for k in _HOLD_EVIDENCE_KEYS):
        verdict = verification.three_predicate_hold(
            gripper_position_rad=float(metrics["gripper_position_rad"]),
            ee_pos_start=tuple(metrics["ee_pos_start"]),
            ee_pos_end=tuple(metrics["ee_pos_end"]),
            object_pos_start=tuple(metrics["object_pos_start"]),
            object_pos_end=tuple(metrics["object_pos_end"]),
            object_rise_m=float(metrics["object_rise_m"]),
            min_lift_m=THRESHOLDS.min_lift_m,
        )
        if not verdict.held:
            failed = [
                why
                for why, ok in (
                    ("gripper not in cage band", verdict.gripper_in_cage_band),
                    ("object did not follow the EE", verdict.object_follows_ee),
                    ("object did not rise", verdict.object_lifted),
                )
                if not ok
            ]
            return SkillOutcome.WEAK_GRASP, "; ".join(failed)
    return SkillOutcome.SUCCESS, ""


def classify_scorer(scored: bool, metrics: dict) -> tuple[SkillOutcome, str]:
    """Generic 'did the stage scorer accept this?' classifier."""
    if scored:
        return SkillOutcome.SUCCESS, ""
    return SkillOutcome.UNSCORED, metrics.get(
        "reason", "scorer predicate not met"
    )


CLASSIFIERS = {
    "navigate": classify_navigate,
    "reach": classify_reach,
    "grasp": classify_grasp,
    "lift": classify_lift,
    "hold": classify_hold,
}


def classify(skill: str, metrics: dict) -> tuple[SkillOutcome, str]:
    """Dispatch to the right classifier; default to scorer-style."""
    fn = CLASSIFIERS.get(skill)
    if fn is not None:
        return fn(metrics)
    return classify_scorer(bool(metrics.get("scored", False)), metrics)
