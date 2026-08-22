# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REV12 T5: candidates must survive a live scene.

A `GraspCandidate.position` is a static absolute world XYZ, recorded at
whatever moment a candidate source (ER, the proven-height injection)
last observed the object. Nothing has ever re-validated it against where
the object ACTUALLY is at attempt time -- and this project has real
physical evidence that objects move: a failed grasp knocked plate2 off
the counter (z 0.7472 -> 0.0044), the cup fell untouched during bowl2's
episode, spoon2 once ended 223m from the gripper
(plans/SYNC.md / ACTIVE_BRIEF PHYSICAL REALITIES).

This module re-anchors a candidate to the object's LIVE observed pose
before each attempt, instead of trusting the absolute position frozen at
candidate-generation time. No schema change needed: `CandidateFile`
already records `object_pose` (the pose at generation time) alongside
each candidate's absolute `position` -- the object-relative OFFSET is
just `candidate.position - object_pose`, computed here, not stored. This
keeps old files (and every existing consumer of `GraspCandidate.position`
as an absolute point) working unchanged -- "object-relative" is a
computation over the existing fields, not a new file format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# If the object moved further than this in XY since the candidate was
# generated, re-anchoring is still attempted (translate rigidly) -- this
# threshold only distinguishes "routine drift" from "notable drift" for
# logging/diagnostics, it does not itself gate whether to proceed.
ROUTINE_DRIFT_XY_M = 0.05

# z dropped more than this since generation -> the object is very likely
# on the floor (knocked off), not sitting on a work surface -- abandon
# rather than spend ~1200 IK ticks discovering the target is unreachable
# air above where the object used to be.
FLOOR_DROP_Z_M = 0.10

# A translation larger than this in any axis is not "the object moved",
# it's a simulation glitch (the 223m spoon2 fling) -- reject rather than
# rigidly translate a candidate by an absurd amount.
MAX_SANE_DELTA_M = 1.0

# Generous scene bounds, derived from grading.py's TASK3_KITCHEN_AREA
# (x in [-5.8,-2.6], y in [-3.85,0.25]) union TASK3_DINING_AREA (x in
# [-5.8,0.10], y in [0.2,3.6]), padded by ~1.2m on every side -- this is
# a coarse sanity clamp against garbage poses (NaN-adjacent, off in
# outer space), not a precise room boundary.
SCENE_BOUNDS_X_M = (-7.0, 1.5)
SCENE_BOUNDS_Y_M = (-5.2, 5.0)
SCENE_BOUNDS_Z_M = (-0.5, 3.0)


class ReanchorAction(str, Enum):
    PROCEED = "proceed"
    ABANDON_FLOOR = "abandon_floor"
    ABANDON_OUT_OF_BOUNDS = "abandon_out_of_bounds"
    ABANDON_JUMP = "abandon_jump"


@dataclass(frozen=True)
class ReanchorResult:
    action: ReanchorAction
    translated_position: tuple[float, float, float] | None
    delta_xy_m: float
    delta_z_m: float
    routine_drift: bool
    reason: str


def _in_scene_bounds(pose: tuple[float, float, float]) -> bool:
    x, y, z = pose
    return (
        SCENE_BOUNDS_X_M[0] <= x <= SCENE_BOUNDS_X_M[1]
        and SCENE_BOUNDS_Y_M[0] <= y <= SCENE_BOUNDS_Y_M[1]
        and SCENE_BOUNDS_Z_M[0] <= z <= SCENE_BOUNDS_Z_M[1]
    )


def reanchor_candidate(
    candidate_position: tuple[float, float, float],
    recorded_object_pose: tuple[float, float, float],
    live_object_pose: tuple[float, float, float],
) -> ReanchorResult:
    """Re-anchor one candidate's absolute position to where the object
    ACTUALLY is right now, or abandon it with a clear, specific reason.

    Order of checks matters: bounds/jump rejection happens before floor
    detection, so a genuinely out-of-scene live pose (e.g. the 223m
    fling) is reported as `abandon_jump`/`abandon_out_of_bounds`, not
    misdiagnosed as a floor drop just because its z also happens to be
    low.
    """
    if not _in_scene_bounds(live_object_pose):
        return ReanchorResult(
            action=ReanchorAction.ABANDON_OUT_OF_BOUNDS,
            translated_position=None,
            delta_xy_m=math.hypot(
                live_object_pose[0] - recorded_object_pose[0],
                live_object_pose[1] - recorded_object_pose[1],
            ),
            delta_z_m=live_object_pose[2] - recorded_object_pose[2],
            routine_drift=False,
            reason=(
                f"live object pose {live_object_pose} is outside scene "
                f"bounds (x={SCENE_BOUNDS_X_M}, y={SCENE_BOUNDS_Y_M}, "
                f"z={SCENE_BOUNDS_Z_M})"
            ),
        )

    delta = tuple(
        live - recorded
        for live, recorded in zip(live_object_pose, recorded_object_pose)
    )
    delta_xy_m = math.hypot(delta[0], delta[1])
    delta_z_m = delta[2]

    if delta_xy_m > MAX_SANE_DELTA_M or abs(delta_z_m) > MAX_SANE_DELTA_M:
        return ReanchorResult(
            action=ReanchorAction.ABANDON_JUMP,
            translated_position=None,
            delta_xy_m=delta_xy_m,
            delta_z_m=delta_z_m,
            routine_drift=False,
            reason=(
                f"object moved {delta_xy_m:.3f}m XY / {delta_z_m:.3f}m Z "
                f"since candidate generation -- exceeds "
                f"{MAX_SANE_DELTA_M}m sanity ceiling, not a real object move"
            ),
        )

    if delta_z_m < -FLOOR_DROP_Z_M:
        return ReanchorResult(
            action=ReanchorAction.ABANDON_FLOOR,
            translated_position=None,
            delta_xy_m=delta_xy_m,
            delta_z_m=delta_z_m,
            routine_drift=False,
            reason=(
                f"object z dropped {-delta_z_m:.3f}m since candidate "
                f"generation (> {FLOOR_DROP_Z_M}m) -- likely on the floor"
            ),
        )

    translated = tuple(cand + d for cand, d in zip(candidate_position, delta))
    return ReanchorResult(
        action=ReanchorAction.PROCEED,
        translated_position=translated,
        delta_xy_m=delta_xy_m,
        delta_z_m=delta_z_m,
        routine_drift=delta_xy_m <= ROUTINE_DRIFT_XY_M,
        reason="",
    )
