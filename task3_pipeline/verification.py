# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REV19 P1.1: one module for hold verification, every predicate pure and
CPU-testable, built on the vendored official grader (`official_scoring.py`)
plus this repo's own already-proven hold predicates.

Does NOT replace `task3_autonomy.grasp_transport`'s existing
`independent_signals`/`verified_close` scheme (an ">= 2 of 4 signals agree"
check that stages.py and other callers already depend on and have tuned
against real GPU data). This module adds a STRICTER, additive check for
AUTONOMOUS_MODE: `three_predicate_hold` requires all three of gripper angle,
delta-follow, and lift -- no partial credit, per REV19's explicit ask ("all
three required"). Existing callers are unaffected until they opt in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from task3_autonomy.grasp_transport import (
    GRIP_QUALITY_CAGED_MAX_RAD,
    GRIP_QUALITY_CAGED_MIN_RAD,
)

DEFAULT_MIN_EE_DELTA_M = 0.02
DEFAULT_FOLLOW_RATIO_TOLERANCE = 0.3


def _distance(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def object_follows_ee_delta(
    ee_pos_start: tuple[float, float, float],
    ee_pos_end: tuple[float, float, float],
    object_pos_start: tuple[float, float, float],
    object_pos_end: tuple[float, float, float],
    *,
    min_ee_delta_m: float = DEFAULT_MIN_EE_DELTA_M,
    ratio_tolerance: float = DEFAULT_FOLLOW_RATIO_TOLERANCE,
) -> bool:
    """The delta-follow check REV19 P1.1 asks for: the end effector moves a
    commanded Delta over N ticks, and the object moves ~Delta too.

    Distinct from `grasp_transport.object_follows_end_effector`, which only
    checks that the object is CLOSE to the end effector at one instant -- a
    static object resting near a stationary gripper would pass that check
    without ever being held. This requires the END EFFECTOR to have moved a
    real amount (`min_ee_delta_m`) before it will call anything a "follow" at
    all, then requires the object's own displacement over the SAME window to
    be within `ratio_tolerance` of the end effector's.
    """
    ee_delta = _distance(ee_pos_start, ee_pos_end)
    if ee_delta < min_ee_delta_m:
        return False
    object_delta = _distance(object_pos_start, object_pos_end)
    return abs(object_delta - ee_delta) <= ratio_tolerance * ee_delta


@dataclass(frozen=True)
class HoldVerdict:
    gripper_in_cage_band: bool
    object_follows_ee: bool
    object_lifted: bool

    @property
    def held(self) -> bool:
        return (
            self.gripper_in_cage_band
            and self.object_follows_ee
            and self.object_lifted
        )


def three_predicate_hold(
    *,
    gripper_position_rad: float,
    ee_pos_start: tuple[float, float, float],
    ee_pos_end: tuple[float, float, float],
    object_pos_start: tuple[float, float, float],
    object_pos_end: tuple[float, float, float],
    object_rise_m: float,
    min_lift_m: float,
    cage_min_rad: float = GRIP_QUALITY_CAGED_MIN_RAD,
    cage_max_rad: float = GRIP_QUALITY_CAGED_MAX_RAD,
    min_ee_delta_m: float = DEFAULT_MIN_EE_DELTA_M,
    follow_ratio_tolerance: float = DEFAULT_FOLLOW_RATIO_TOLERANCE,
) -> HoldVerdict:
    """REV19 P1.1: gripper angle in the profile's cage band AND
    object-follows-EE over a commanded delta AND object rose >= min_lift_m
    -- ALL THREE required, no partial credit. Returns every predicate's
    individual result so a failure is diagnosable, not just a bool.
    """
    gripper_ok = cage_min_rad <= gripper_position_rad <= cage_max_rad
    follows_ok = object_follows_ee_delta(
        ee_pos_start,
        ee_pos_end,
        object_pos_start,
        object_pos_end,
        min_ee_delta_m=min_ee_delta_m,
        ratio_tolerance=follow_ratio_tolerance,
    )
    lift_ok = object_rise_m >= min_lift_m
    return HoldVerdict(
        gripper_in_cage_band=gripper_ok,
        object_follows_ee=follows_ok,
        object_lifted=lift_ok,
    )
