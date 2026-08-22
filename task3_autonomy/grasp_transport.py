# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared grasp/hold verification thresholds and the verified_* result
contract (plans/handoff.md sec 16.5/16.6).

Before this module existed, ``classify_grip_quality``/``independent_signals``
and the caged-band/hold-distance constants were defined three separate times
(``run_stage1_setup.py``, ``run_stage4_cleanup.py``,
``run_stage2_feeding.py``), each with its own copy -- and the copies had
already drifted (sec 16.2 item 4: ``run_stage4_cleanup.py`` still carried the
old 0.5m ``HOLD_MAX_DISTANCE_M`` instead of the corrected 0.15m). This module
is the one place those values and that classification logic are defined;
every stage script should import from here instead of declaring them again.

These are VERIFICATION thresholds -- they decide whether we believe a grasp
happened -- not physical parameters. Nothing here changes gripper geometry,
offsets, servo gains, or physics (sec 13.6/15.6 explicitly distinguishes this
from the closed grasp-parameter-tuning question).
"""

from __future__ import annotations

import math

# --- Approach kinematics (plans/handoff.md sec 4.42/16.6) ------------------
# Owner correction 2026-07-26 (sec 4.42): sec 16.6 moved the VERIFICATION
# vocabulary (below) into this module but left the APPROACH geometry
# duplicated in run_stage4_cleanup.py and never ported into
# run_stage1_setup.py at all -- that omission, not the close-phase regression
# or the retry mechanism, is the traced root cause of Q6's 0/13. Both stage
# scripts must import the grasp heading/stance from here; neither may keep
# its own copy.
CORRIDOR_STOP = (-3.18, -1.6)
# L3 (2026-08-02, SYNC 18/19): (-3.0,-3.1) sits only 0.033m (real, USD
# bbox-measured) from a real obstacle, inside the base's own 0.40m
# half-width need -- moved to (-3.3,-3.1), which has >=0.226m of real
# margin from every real floor-level obstacle within 0.4m, including
# KITCHEN_ISLAND_BBOX. See scripts/task3/verify_grasp_lift.py's
# ROTATE_SPOT for the full evidence trail (this constant duplicates that
# one's value, per this file's own header comment above about both stage
# scripts importing from here).
ROTATE_SPOT = (-3.3, -3.1)
# East-stance approach: robot at island east face, facing west. Measured
# (scripts/task3/measure_object_geometry.py, real USD bounding-box query):
# ISLAND_STANCE is the cup's own live position plus STANCE_OFFSET_EAST below
# (matches to within 1mm: cup center [-4.1849,-1.7522] + (0.865,0.033) =
# [-3.3199,-1.7192] vs ISLAND_STANCE [-3.32,-1.72]).
ISLAND_STANCE = (-3.32, -1.72)
FACE_WEST_YAW_RAD = math.pi
# bowl2/plate2/spoon2 spawn at DIFFERENT positions on the same counter
# (bowl2 [-4.2985,-1.4999], plate2 [-4.3087,-1.6609], spoon2
# [-4.38,-1.6712]) -- applying the fixed cup-tuned ISLAND_STANCE to them
# puts the base 9-25cm away from where this same formula says it should be
# (sec 4.42 Finding 2, originally measured in run_stage4_cleanup.py).
STANCE_OFFSET_EAST = (0.865, 0.033)
# Full-route wheel settling ends about 3 degrees clockwise of its commanded
# heading. The frozen physical cup-grasp baseline approached at 3.098 rad,
# so command a small counter-clockwise bias before the final east stance to
# reproduce its measured jaw/rim geometry (rather than changing the cup
# target or any rigid-body property). Measured for cup only (sec 4.42
# Finding 1) -- 0.095 rad at the cup's ~0.865m dead-ahead reach is ~8.2cm of
# lateral jaw displacement, roughly one cup diameter, which is the whole
# 0.043-0.052 rim-catch vs 0.0709-0.076 proven-cage gap in sec 15.
EAST_CUP_GRASP_HEADING_BIAS_RAD = -0.095


def stance_for_object(
    object_name: str, object_position: tuple[float, float, float]
) -> tuple[float, float]:
    """East-approach stance: the frozen literal for cup, live-derived for
    everything else (see STANCE_OFFSET_EAST comment above)."""
    if object_name == "cup":
        return ISLAND_STANCE
    return (
        object_position[0] + STANCE_OFFSET_EAST[0],
        object_position[1] + STANCE_OFFSET_EAST[1],
    )


def grasp_heading_rad(
    object_name: str,
    *,
    heading_bias_rad: float = EAST_CUP_GRASP_HEADING_BIAS_RAD,
) -> float:
    """The base yaw to rotate to before the final stance approach.

    Sec 4.42: only the cup has a measured heading-bias correction. Do not
    silently extend it to bowl2/plate2/spoon2 -- they get the uncorrected
    FACE_WEST_YAW_RAD until each object's own bias is separately measured.
    """
    if object_name == "cup":
        return FACE_WEST_YAW_RAD + heading_bias_rad
    return FACE_WEST_YAW_RAD


# Sec 15.3's three-regime model, derived from proven vs. failed close-phase
# gripper_position_rad across every run recorded in handoff.md sec 4: frozen
# proof 0.0709, run18 0.076 (both caged + scored); all 13 2026-07-26 attempts
# 0.0433-0.0521 (marginal, 0/13 against this band); grasp_optimization archive
# 0.0007/0.0081 (air -- cup squirted out before the jaws met).
GRIP_QUALITY_CAGED_MIN_RAD = 0.065
# Sec 15.3 specifies the caged band as 0.065-0.09, not 0.065-unbounded. A prior
# unbounded upper check mislabeled a wide-open gripper (1.0249/0.9885 rad, near
# GRIPPER_OPEN_RAD=0.9) as "caged", when it is the opposite -- never really
# closed at all (sec 4.29 Finding 2).
GRIP_QUALITY_CAGED_MAX_RAD = 0.09
GRIP_QUALITY_MARGINAL_MIN_RAD = 0.02
CONTACT_FORCE_MIN_N = 0.01

# Sec 17.4 #3 (P0.5): the band above is CUP's proven cage, not a universal
# constant -- applying it to every object was wrong and is why Stage-4
# breadth read 0/3 (sec 4.49): bowl2 closed to a real but wider 0.205 rad
# and was scored identically to a miss. Objects not listed here fall back to
# the cup band, so every existing caller that doesn't pass ``object_name``
# keeps behaving exactly as before (the frozen §10 cup path's requirement).
OBJECT_GRIP_BANDS: dict[str, tuple[float, float]] = {
    "cup": (GRIP_QUALITY_CAGED_MIN_RAD, GRIP_QUALITY_CAGED_MAX_RAD),
    # [INFERRED, pending GPU confirmation -- handoff.md sec 17.6 Phase-1
    # Worker D] centered on the one real bowl2 close observed so far (sec
    # 4.49: 0.205 rad, "a real, if loose... partial closure around
    # something, not a miss"). Not yet proven by a scored lift+hold.
    "bowl2": (0.15, 0.25),
}

# spoon2 (thin-handle pinch) and plate2 (rim/edge grasp) both approach fully
# closed on a real grip AND on a miss -- sec 4.49 measured spoon2 closing to
# -0.0 rad on nothing and plate2 to 1.0267 rad (essentially open) also on
# nothing. Angle cannot discriminate grip from air for either object, so
# they classify on measured finger contact force instead; angle is not
# consulted at all for them (sec 17.4 #3: "angle secondary").
CONTACT_CLASSIFIED_OBJECTS = frozenset({"spoon2", "plate2"})

# The object must rise at least this much off its start height to count as
# lifted (sec 15.6).
MIN_LIFT_M = 0.05

# object_follows_end_effector's own sensible default, matching the frozen
# proof's measured object_to_ee_m of 0.1483m (sec 15.2). The legacy 0.5m value
# (3.4x the real measured distance) let a real drop go undetected -- sec
# 4.21's "held for the full 90s" was never actually confirmed against it.
DEFAULT_HOLD_MAX_DISTANCE_M = 0.15
# Kept only so callers can reproduce old (broken) runs for comparison against
# fresh evidence -- never use this as a default.
LEGACY_HOLD_MAX_DISTANCE_M = 0.5


def classify_grip_quality(
    gripper_position_rad: float,
    object_name: str = "cup",
    *,
    contact_force_n: float | None = None,
) -> str:
    """Grade the grip continuously instead of thresholding it (sec 15.6 V2).

    Does not change any close/accept gate -- this is an additional
    diagnostic label, computed at the harness level.

    ``object_name`` selects the angle band (sec 17.4 #3, P0.5): defaults to
    "cup" so every existing caller that passes only the angle is completely
    unaffected -- this is the frozen §10 cup path's own band. For
    ``CONTACT_CLASSIFIED_OBJECTS`` (spoon2, plate2) the angle is ignored
    entirely and the verdict comes from ``contact_force_n`` instead, since a
    real grip and a miss both approach fully-closed for those objects.
    """
    if object_name in CONTACT_CLASSIFIED_OBJECTS:
        if contact_force_n is None:
            return "unknown"
        return "caged" if contact_force_n > CONTACT_FORCE_MIN_N else "air"
    min_rad, max_rad = OBJECT_GRIP_BANDS.get(
        object_name, (GRIP_QUALITY_CAGED_MIN_RAD, GRIP_QUALITY_CAGED_MAX_RAD)
    )
    if gripper_position_rad > max_rad:
        return "open"
    if gripper_position_rad >= min_rad:
        return "caged"
    if gripper_position_rad >= GRIP_QUALITY_MARGINAL_MIN_RAD:
        return "marginal"
    return "air"


def independent_signals(
    *,
    gripper_position_rad: float | None = None,
    contact_force_n: float | None = None,
    object_rose: bool | None = None,
    object_follows_ee: bool | None = None,
) -> list[str]:
    """Return the names of the independent evidence signals that hold true.

    Sec 15.6 V5: a phase may only report ok:true if >=2 of these agree. Each
    kwarg is None when that signal was not measured this run (e.g.
    contact_force_n is None when no contact sensor was wired in) -- None is
    never counted as agreeing, it is simply absent from the returned list.
    """
    signals: list[str] = []
    if (
        gripper_position_rad is not None
        and GRIP_QUALITY_CAGED_MIN_RAD
        <= gripper_position_rad
        <= GRIP_QUALITY_CAGED_MAX_RAD
    ):
        signals.append("gripper_rad")
    if contact_force_n is not None and contact_force_n > CONTACT_FORCE_MIN_N:
        signals.append("contact_force_n")
    if object_rose:
        signals.append("object_rose")
    if object_follows_ee:
        signals.append("object_follows_ee")
    return signals


def object_follows_end_effector(
    object_position: tuple[float, float, float],
    end_effector_position: tuple[float, float, float],
    max_distance_m: float = DEFAULT_HOLD_MAX_DISTANCE_M,
) -> bool:
    """Is the object still within max_distance_m of the end effector.

    Owner correction 2026-07-26 (sec 4.38 amendment item 7): moved here from
    a per-script closure so the 0.15m boundary itself is unit-testable
    (below/at/above), not just exercised implicitly inside a GPU run.
    """
    dx = object_position[0] - end_effector_position[0]
    dy = object_position[1] - end_effector_position[1]
    dz = object_position[2] - end_effector_position[2]
    distance = (dx * dx + dy * dy + dz * dz) ** 0.5
    return distance <= max_distance_m


def contact_force_state(contact_force_n: float | None) -> str:
    """Sec 4.38 amendment item 7: a missing sensor and a genuine zero
    reading are different physical states and must never be conflated --
    sec 4.37 already got confused by exactly this once. Returns one of
    "unavailable" (no contact sensor wired this run), "below_threshold"
    (sensor wired, reading <= CONTACT_FORCE_MIN_N), or "detected"."""
    if contact_force_n is None:
        return "unavailable"
    if contact_force_n <= CONTACT_FORCE_MIN_N:
        return "below_threshold"
    return "detected"


def verified_close(grip_quality: str) -> bool:
    """Sec 16.5: verified_close requires the close to have landed in the
    caged band -- not merely that the mechanical close command completed
    (that raw boolean is ``mechanical_close_ok``, kept separately since sec
    10's frozen Stage-4 proof reads the same field and must not change
    meaning)."""
    return grip_quality == "caged"


def close_failure_reason(grip_quality: str) -> str | None:
    """None when verified_close would be True; otherwise a short, stable
    string naming which of the failing regimes this was (sec 4.38 amendment
    item 7's required ``failure_reason`` field). ``"unknown"`` is P0.5's
    addition: a contact-classified object (spoon2/plate2) with no contact
    force reading available -- angle can't help here either, so this is
    genuinely unclassifiable, not a specific failure mode."""
    if grip_quality == "caged":
        return None
    return {
        "air": "close_air_no_contact",
        "marginal": "close_marginal_below_caged_band",
        "open": "close_wide_open_never_closed",
        "unknown": "close_unclassifiable_no_contact_sensor",
    }[grip_quality]


def verified_lift(object_rose: bool) -> bool:
    """Sec 4.38 amendment item 7: the lift phase's own verified_* field,
    separate from verified_hold (which additionally requires the hold-window
    tracking + force/angle evidence). True exactly when the object cleared
    MIN_LIFT_M off its start height."""
    return bool(object_rose)


def verified_hold(
    *,
    object_rose: bool,
    object_follows_ee: bool,
    grasp_force_n: float | None,
    gripper_position_rad: float,
    object_name: str = "cup",
) -> bool:
    """Sec 16.5's >=2-signal V5 rule, sampled at the END of the hold window,
    not the start: object rose, AND still follows the end effector, AND
    (measured contact force OR the gripper angle is still in the object's
    caged band). ``object_name`` defaults to "cup" (P0.5, sec 17.4 #3) so
    every existing caller keeps the exact frozen §10 band unless it opts in;
    it only changes the angle *fallback* -- a wired contact-force reading
    still wins regardless of object."""
    min_rad, max_rad = OBJECT_GRIP_BANDS.get(
        object_name, (GRIP_QUALITY_CAGED_MIN_RAD, GRIP_QUALITY_CAGED_MAX_RAD)
    )
    force_or_caged = (
        grasp_force_n is not None and grasp_force_n > CONTACT_FORCE_MIN_N
    ) or (min_rad <= gripper_position_rad <= max_rad)
    return bool(object_rose and object_follows_ee and force_or_caged)


def hold_failure_reason(
    *,
    object_rose: bool,
    object_follows_ee: bool,
    grasp_force_n: float | None,
    gripper_position_rad: float,
    object_name: str = "cup",
) -> str | None:
    """None when verified_hold would be True; otherwise names the first
    failing signal, checked in the same order sec 15.6 V5 evaluates them."""
    if verified_hold(
        object_rose=object_rose,
        object_follows_ee=object_follows_ee,
        grasp_force_n=grasp_force_n,
        gripper_position_rad=gripper_position_rad,
        object_name=object_name,
    ):
        return None
    if not object_rose:
        return "hold_object_did_not_rise"
    if not object_follows_ee:
        return "hold_object_separated_from_ee"
    return "hold_lost_grip_signal"


# Same >=2-signal contract as verified_hold, sampled at the end of the carry
# (transit) window instead of the end of the initial hold -- kept as a
# separate name per sec 16.5 so log entries record which phase produced the
# verdict, even though the evidence rule is identical.
verified_carry = verified_hold


# Default "did an approach move that wasn't supposed to touch the object yet
# actually touch it" tolerance -- distinct from DEFAULT_HOLD_MAX_DISTANCE_M
# (object-to-EE distance during an intended hold). Generic across objects; a
# real approach/descend should leave the object within a couple cm of where
# it started until the intentional close.
OBJECT_INTEGRITY_TOL_M = 0.03


def object_disturbed(
    start_xyz: tuple[float, float, float],
    live_xyz: tuple[float, float, float],
    tol_m: float = OBJECT_INTEGRITY_TOL_M,
) -> bool:
    """True when an object has moved more than ``tol_m`` from ``start_xyz``.

    Sec 17 (P0.4): the shared generalisation of the Stage-1 0.31m cup
    displacement (sec 17.4 #1) and the Stage-2 defect where the spoon was
    knocked off the island mid-descend and the harness closed on empty floor
    anyway (sec 17.4 #2). Call this after every arm movement in Stage 1 and
    Stage 2 that is not itself supposed to move the object, and fail fast
    rather than continuing toward a close on an object that already moved.
    """
    dx = live_xyz[0] - start_xyz[0]
    dy = live_xyz[1] - start_xyz[1]
    dz = live_xyz[2] - start_xyz[2]
    distance = (dx * dx + dy * dy + dz * dz) ** 0.5
    return distance > tol_m


def verified_placement(scored: bool) -> bool:
    """Sec 16.5: placement is verified exactly when the real scorer
    (``score_stage1_table_setup`` et al.) counted the object -- there is no
    separate placement heuristic to duplicate here."""
    return bool(scored)
