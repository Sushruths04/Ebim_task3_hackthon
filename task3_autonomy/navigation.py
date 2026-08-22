# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure navigation math for the Task 3 FSM's navigate_to() skill.

No Isaac Sim imports here -- unit-testable on CPU (docs/task3_master_plan.md
hard rule 5: write and unit-test FSM/skill code locally before touching a
GPU). The Isaac-dependent orchestration (reading live robot pose, calling
tmr_base_control.compute_drive_targets()/compensate_yaw_rate(), stepping the
sim) lives in task3_autonomy/skills.py and reuses the functions below.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

# The measured arm reach ceiling, imported rather than retyped -- it caps
# how far `_rotate_to_clear_island` may grow a stance radius when the
# requested one is geometrically impossible. Safe both ways:
# `perception_targets`'s module level is stdlib-only (its `omni.usd`/`pxr`
# imports live inside a function) and it does not import this module, so
# this file stays CPU-testable with no Isaac import.
from task3_autonomy.perception_targets import (
    REACH_LIMIT_M as MEASURED_REACH_LIMIT_M,
)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float  # radians


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def waypoints_y_then_x(
    start_xy: tuple[float, float], target_xy: tuple[float, float]
) -> list[tuple[float, float]]:
    """Route the y move before the x move.

    Mirrors the routing idea in
    scripts/evaluation/task3/integration_test.py's y_then_x_path(), which
    exists to avoid the kitchen/dining wall partition -- that helper drives
    a kinematic prim path for grading tests; this one produces the waypoint
    list a real closed-loop base controller drives through.
    """
    midpoint = (start_xy[0], target_xy[1])
    waypoints = [start_xy]
    if midpoint != waypoints[-1]:
        waypoints.append(midpoint)
    if target_xy != waypoints[-1]:
        waypoints.append(target_xy)
    return waypoints


# Task 3 room geometry, measured from assets/robot_room.usd (2026-07-17):
# the dining/kitchen partition runs along y in [0.10, 0.34] with a single
# doorway gap x in (-4.74, -3.54); the kitchen island (grading-prop counter)
# occupies x [-4.51, -3.77] x y [-2.47, -1.22]. The first live runs proved
# walls are load-bearing: a y-then-x route descending at x=-4.60 hugged the
# door jamb (jamb edge x=-4.74) and stalled the base at y ~ 0.99 with wheels
# contact-stalled (NAVDBG, /tmp/task3_verify_nav7.log on sim-dev-g4b).

TASK3_DOOR_X = -4.14
TASK3_DOOR_Y = 0.22
# rev 12 T4: raised from 0.9 -- two independent, bit-identical GPU repros
# (navigate_only_crossing_test.py, seats D and F, different seeds) both
# deterministically stalled at (-3.769, 1.036) immediately after crossing,
# while route_via_door's post-crossing traverse holds y at north_point's
# y (TASK3_DOOR_Y + this constant). assets/derived/room_obstacles.json has
# NO registered obstacle anywhere in that corridor (x in [-4.3,-1.5], y in
# [0.7,1.5], z<1.2), so whatever is there is not in the static room scan
# (a dynamic prop such as a dining chair is the leading hypothesis, not
# confirmed). Raising the traverse row to y~1.42 stays inside that same
# verified-empty band while moving ~0.4m clear of the measured freeze
# point -- plans/SYNC.md rev 12 T3/T4.
TASK3_DOOR_APPROACH_M = 1.2

# East edge of the doorway gap. The partition is solid for x > -3.54;
# the base must be at or west of this x to pass through the gap.
# Measured from assets/robot_room.usd (2026-07-17).
TASK3_GAP_EAST_EDGE = -3.54

# Kitchen-side door point sits in the shallow lane between the partition
# (south face y=0.10) and the kitchen island (north face y=-1.22): rear
# extent ~0.42 clears the wall (rear tip 0.05) and the tucked arms' 0.80 m
# effective forward overhang clears the island (nose tip -1.17) — ~5 cm
# margin each way. Live evidence: nav9 scraped the island for ~35 s with
# the 0.885 m overhang of the v3 fold at lane y=-0.68.
TASK3_KITCHEN_LANE_Y = -0.37

# Full kitchen island bbox (same measurement as the doorway geometry above,
# assets/robot_room.usd, 2026-07-17) and the base's half-width, used to keep
# any navigate/stance target clear of the island footprint (ACTIVE_BRIEF.md
# sec 3: every Stage 1/4 stall traced to a navigate target sitting inside
# this box).
KITCHEN_ISLAND_BBOX = dict(x_min=-4.51, x_max=-3.77, y_min=-2.47, y_max=-1.22)

# The room's real west wall. CORRECTED 2026-08-14 -- the previous value
# (`x_min=-5.947, x_max=-4.067, y_min=-4.040, y_max=0.771`) was the whole-
# prim AABB of `/root/root/Line041/Line041`, and it was wrong by 1.64 m in
# x. Per-face decomposition of that prim (`scripts/task3/
# generate_room_obstacles.py`, run against the real USD on GPU) shows
# Line041 is a 20 mm COPING STRIP sitting on top of the walls at
# `z in [1.177, 1.197]` -- its faces trace the wall TOPS (x[-5.947,-5.707]
# west, y[-4.040,-3.800] south, y[0.101,0.341] north alcove), and the AABB
# enclosing that L-shaped run is mostly the open floor between the arms of
# the L. `generate_room_obstacles.py`'s own note ("a real collider at
# ~1.18m world z, arm/held-object height not base height") was right all
# along; this constant hand-typed it back in as a base obstacle anyway.
#
# The real wall at base height is `/root/root/Rectangle006/Rectangle006`,
# `z in [-0.003, 1.177]`, a 0.24 m slab whose inner face is at x=-5.707.
#
# What the bad value cost, measured: it declared the ENTIRE west third of
# the room solid, including the kitchen counter itself, all four Stage-1/4
# scoring objects, `route_via_door`'s own driven kitchen-lane waypoint
# (-4.14, -0.37), and the base pose (-4.28, -0.82) a real 2026-08-14
# episode physically drove to. `_rotate_to_clear_island` therefore had
# ZERO clearing candidates for every kitchen object (0 of 180 sweep
# angles), exhausted its sweep, and silently returned the un-cleared
# nominal stance -- a point inside the counter's own inflated footprint.
# That, not the drive, is the "base-drive dead zone at x~-4.3 to -4.5".
WEST_WALL_BBOX = dict(x_min=-5.947, x_max=-5.707, y_min=-3.790, y_max=0.771)

BASE_HALF_WIDTH = 0.40

# Mirrors task3_autonomy/skills.py's WAYPOINT_PASS_TOLERANCE_M (0.10m) --
# how far short of an intermediate waypoint NavigateTo.compute() will
# already treat it as "passed" and start heading to the next one. Kept
# as a separate constant (not imported) because skills.py imports FROM
# this module; used only to widen the kitchen-descent corner in
# route_via_door, see its comment there for the full mechanism.
_KITCHEN_CORNER_CUT_BUFFER_M = 0.10

# A bbox only blocks the BASE if its z-span overlaps the band the robot's
# body actually occupies. `nearest_obstacle_clearance_m` was z-agnostic,
# so a 2 mm floor rug and a ceiling light counted the same as a wall --
# 93 of the 289 generated boxes (32%) are one or the other, and one of the
# rugs sits directly under `TASK_ROBOT_POSES["task3"]`, making the robot's
# own spawn read as "inside an obstacle".
#
# Lower bound: the asset authors its wheel contact plane at z=0 (see
# `scripts/task3/probe_base_collider_extents.py`), so anything entirely
# below z=0 is a decal painted on the floor, not something to route round.
# Upper bound: the robot's tallest structure is the head camera at ~1.35 m
# (read off live TF, see `scripts/task3/derive_standoff_pose.py`) with the
# spine adding at most its 0.45 m travel; the asset's structural walls top
# out at 1.197 m. 2.0 m clears everything the robot can occupy and still
# sits below every overhead fixture in the asset (lowest is 2.517 m).
BASE_BLOCKING_Z_MIN_M = 0.0
BASE_BLOCKING_Z_MAX_M = 2.0

# The proven reach-safe base-relative offset from an object's live xy
# (STANCE - cup_position from verify_grasp_lift.py's 10/10 cup grasp).
# Its magnitude is the arm's reach budget; only the direction/placement
# is adjusted per-object by stance_for() below.
#
# S2 sub-fix 3 (plans/LOOP_PROMPT_VM_A.md rev 2, P4; root cause: GATE C2.5A,
# curobo_stance.py sec 64): the original (0.865, 0.033) magnitude is
# ~0.866m -- PAST the ~0.855m measured reach ceiling (GATE S1). This
# function is `curobo_stance_for()`'s fallback whenever cuRobo finds no
# valid candidate at all, so the original value made that fallback a
# guaranteed-unreachable stance by construction, not merely a degraded
# one. Scaled to keep the same approach direction while bringing the
# magnitude to ~0.78m, matching sub-fix 2's annulus cap so both fixes
# leave the same real margin against the same measured ceiling.
STANCE_OFFSET_EAST = (0.7794, 0.0297)  # (dx, dy) from live object xy
STANCE_YAW_EAST_RAD = math.pi  # face west
STANCE_REACH_RADIUS_M = math.hypot(*STANCE_OFFSET_EAST)


def waypoints_x_then_y(
    start_xy: tuple[float, float], target_xy: tuple[float, float]
) -> list[tuple[float, float]]:
    """Route the x move before the y move (mirror of waypoints_y_then_x).

    Needed on the kitchen side: descending at the door's x runs into the
    kitchen island, so cross the clear row first, then descend.
    """
    midpoint = (target_xy[0], start_xy[1])
    waypoints = [start_xy]
    if midpoint != waypoints[-1]:
        waypoints.append(midpoint)
    if target_xy != waypoints[-1]:
        waypoints.append(target_xy)
    return waypoints


def route_via_door(
    start_xy: tuple[float, float], target_xy: tuple[float, float]
) -> list[tuple[float, float]]:
    """Waypoint route that crosses the partition only through the doorway.

    Same-side start/target fall back to plain y-then-x. Crossing routes:
    y-then-x to the door point on the start side (dining side is open at
    +TASK3_DOOR_APPROACH_M; kitchen side uses the shallow
    TASK3_KITCHEN_LANE_Y lane), straight through the gap, then x-then-y on
    the far side (hugs the wall to clear the island).
    """
    start_north = start_xy[1] > TASK3_DOOR_Y
    target_north = target_xy[1] > TASK3_DOOR_Y
    if start_north == target_north:
        return waypoints_y_then_x(start_xy, target_xy)
    north_point = (TASK3_DOOR_X, TASK3_DOOR_Y + TASK3_DOOR_APPROACH_M)
    south_point = (TASK3_DOOR_X, TASK3_KITCHEN_LANE_Y)
    first, second = (
        (north_point, south_point)
        if start_north
        else (south_point, north_point)
    )

    # Kitchen-side start: if the base's west edge is inside the island
    # (island east face x=-3.77, base half-width ≈ 0.40), the y-then-x
    # route drives north first and drags the base's west side along the
    # island edge, pinning the base.  Insert an eastward waypoint to clear
    # the island before proceeding.
    ISLAND_CLEAR_X = (
        KITCHEN_ISLAND_BBOX["x_max"] + BASE_HALF_WIDTH + 0.05
    )  # -3.32
    # Island-clear only needed for the first leg from the kitchen stance.
    # If we are already near the door lane (within 0.3 m of KITCHEN_LANE_Y),
    # skip the island-clear — otherwise we detour east for no reason.
    near_door = abs(start_xy[1] - TASK3_KITCHEN_LANE_Y) < 0.3
    if not start_north and start_xy[0] < ISLAND_CLEAR_X and not near_door:
        clear_xy = (ISLAND_CLEAR_X, start_xy[1])
        route = [start_xy, clear_xy]
        _sub = waypoints_y_then_x(clear_xy, first)
        route.extend(_sub[1:])  # skip clear_xy (already in route)
    else:
        route = waypoints_y_then_x(start_xy, first)
    print(
        f"DEBUG route_via_door start={start_xy} target={target_xy} "
        f"first={first} second={second} route={route}",
        flush=True,
    )
    route.append(second)
    # 2026-08-16: give the kitchen-descent corner extra clearance before
    # committing to the final descent. NavigateTo (skills.py) passes
    # intermediate waypoints loosely -- WAYPOINT_PASS_TOLERANCE_M=0.10m,
    # duplicated below as _KITCHEN_CORNER_CUT_BUFFER_M since importing
    # skills.py here would be circular (skills.py already imports FROM
    # this module) -- so the base can start heading toward the FINAL
    # target before its x has actually converged to this corner's x.
    # When the corner sits close to a real obstacle (any stance hugging
    # the island's west edge does, by construction: a stance only needs
    # BASE_HALF_WIDTH of clearance, not BASE_HALF_WIDTH plus pass-
    # tolerance slop), the base can still be mid-convergence in x when
    # it enters the obstacle's y-span, driving its tucked-arm envelope
    # (reset()'s own comment: TRANSIT_ARM_POSE exists "to keep the
    # tucked arms from sweeping the island counter") into the real
    # collision geometry and physically stalling there. GPU-measured
    # (outputs/keep_verify_navigate_retry_fix_v2.log, cup stance
    # (-4.9608,-1.6736)): the base froze at (-4.896,-0.837), inside the
    # island's BASE_HALF_WIDTH-inflated bbox, and the arm's approach-
    # phase IK target became genuinely unreachable from there -- 47
    # consecutive "Right arm IK failed: solver reported no solution"
    # retries, nothing ever re-drove the base afterward
    # (outputs/keep_capture_robot_cameras_v2.log, pre-existing bug).
    #
    # If the corner's REAL clearance (nearest_obstacle_clearance_m, the
    # GPU-measured generated set -- more accurate than the hand-typed
    # KITCHEN_ISLAND_BBOX) is inside that slop, push the corner further
    # from the obstacle by the deficit so a full corner-cut still lands
    # with real margin. Only the corner moves; the FINAL leg from there
    # to the true target_xy is tracked with NavigateTo's strict final-
    # arrival tolerance (0.03m default), not this loose one, so it does
    # not reintroduce the same risk. This is additive and a no-op
    # (byte-identical route) whenever the corner already clears with
    # margin -- the common case away from any obstacle, and exactly what
    # `test_rotate_spot_leg_takes_no_westward_detour` guards for a
    # DIFFERENT route entirely (this branch is the ONLY caller of
    # `waypoints_x_then_y`, so this cannot affect that route or any
    # other).
    sub_route = waypoints_x_then_y(second, target_xy)
    if len(sub_route) == 3:
        corner_x, corner_y = sub_route[1]
        clearance_m = nearest_obstacle_clearance_m(
            (corner_x, corner_y), _cached_room_obstacles()
        )
        required_m = BASE_HALF_WIDTH + _KITCHEN_CORNER_CUT_BUFFER_M
        deficit_m = required_m - clearance_m
        if deficit_m > 0.0:
            island_cx = (
                KITCHEN_ISLAND_BBOX["x_min"] + KITCHEN_ISLAND_BBOX["x_max"]
            ) / 2.0
            safer_x = (
                corner_x - deficit_m
                if corner_x < island_cx
                else corner_x + deficit_m
            )
            sub_route[1] = (safer_x, corner_y)
            print(
                "KITCHEN_CORNER_WIDENED "
                f"original_corner=({corner_x:.4f},{corner_y:.4f}) "
                f"clearance_m={clearance_m:.4f} required_m={required_m:.4f} "
                f"widened_corner=({safer_x:.4f},{corner_y:.4f})",
                flush=True,
            )
    route.extend(sub_route[1:])
    return route


def base_twist_toward(
    pose: Pose2D,
    target_xy: tuple[float, float],
    *,
    max_linear_mps: float,
    position_kp: float = 1.5,
    min_creep_mps: float = 0.0,
) -> tuple[float, float]:
    """Body-frame (vx, vy) command driving `pose` toward `target_xy`.

    Yaw is not controlled here. The TMR base is omnidirectional (every
    wheel module steers independently), so it does not need to face its
    direction of travel -- combine this with
    tmr_base_control.compensate_yaw_rate() for heading hold instead of
    reimplementing yaw control here.

    L3 (2026-08-02): `linear_speed = position_kp * distance` decays to
    zero as distance shrinks, by design -- but the wheel drives have a
    real, documented, ~2s velocity-tracking lag at their tuned damping
    (`TmrBaseAdapter.DRIVE_DAMMPING=500`, `skills.py`). A commanded speed
    that keeps shrinking faster than the wheels can track it never lets
    the actual velocity catch up to the commanded one -- this is a
    plausible, not-yet-GPU-confirmed root cause for `navigate_rotate_spot`
    consistently stalling ~0.087m short of its target (handoff/SYNC,
    2026-08-02) despite a nominal 3cm tolerance. `min_creep_mps` (0 by
    default -- opt-in, does not change any existing caller unless it
    passes a nonzero value) floors the commanded speed at a value large
    enough to stay above whatever plateau the real drive stalls at,
    instead of asking the wheels to track an ever-decaying target in the
    final approach. Caller stops commanding entirely the instant
    `pose_reached()` is true, so a floor well above the tolerance
    distance's own natural P-term does not risk a meaningful overshoot
    (worst case is 1 tick's travel at `min_creep_mps`, `sim.cfg.dt=0.005s`
    typical => <=1mm).
    """
    dx_world = target_xy[0] - pose.x
    dy_world = target_xy[1] - pose.y
    distance = math.hypot(dx_world, dy_world)
    if distance < 1e-9:
        return 0.0, 0.0

    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    # Rotate the world-frame error vector by -yaw into the body frame.
    vx_body_unit = cos_yaw * dx_world + sin_yaw * dy_world
    vy_body_unit = -sin_yaw * dx_world + cos_yaw * dy_world

    linear_speed = min(max_linear_mps, position_kp * distance)
    if min_creep_mps > 0.0:
        linear_speed = min(max_linear_mps, max(linear_speed, min_creep_mps))
    scale = linear_speed / distance
    return vx_body_unit * scale, vy_body_unit * scale


@dataclass
class ProgressWatchdog:
    """Detects a stalled base/EE loop by sampled displacement, not by budget.

    handoff.md sec 18.1b: every long tick-bounded loop (navigate_to,
    _rotate_to, carry_object_to) previously ran to its full budget and then
    reported only "converged" or "budget exhausted" -- indistinguishable
    from each other without a per-tick position trace. Four sessions
    (sec 4.60/4.63/4.64/4.65) burned full GPU waves re-deriving that
    distinction after the fact. Sampling every `sample_every_ticks` and
    comparing against the sample `stall_window_ticks` back turns a stall
    into a *named* failure (`stalled=True` + `pose_trace`) the moment it is
    detected, instead of only after the whole budget elapses.
    """

    sample_every_ticks: int = 250
    stall_window_ticks: int = 1000
    min_move_m: float = 0.01
    pose_trace: list[tuple[int, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lookback = max(
            1, self.stall_window_ticks // self.sample_every_ticks
        )

    def sample(self, tick: int, x: float, y: float) -> bool:
        """Call once per tick with the current tick index and base/EE xy.

        Returns True the first tick a stall is confirmed: the sampled
        position has moved less than `min_move_m` since the sample
        `stall_window_ticks` ago. Off-schedule ticks (not a multiple of
        `sample_every_ticks`) are free and always return False.
        """
        if tick % self.sample_every_ticks != 0:
            return False
        self.pose_trace.append((tick, round(x, 4), round(y, 4)))
        if len(self.pose_trace) <= self._lookback:
            return False
        _, prev_x, prev_y = self.pose_trace[-self._lookback - 1]
        return math.hypot(x - prev_x, y - prev_y) < self.min_move_m


def pose_reached(
    pose: Pose2D,
    target_xy: tuple[float, float],
    target_yaw: float | None = None,
    *,
    position_tolerance_m: float = 0.03,
    yaw_tolerance_rad: float = math.radians(3.0),
) -> bool:
    """Stop-condition check: ~3 cm / 3 deg tolerance per the master plan."""
    distance = math.hypot(target_xy[0] - pose.x, target_xy[1] - pose.y)
    if distance > position_tolerance_m:
        return False
    if target_yaw is None:
        return True
    return abs(wrap_to_pi(target_yaw - pose.yaw)) <= yaw_tolerance_rad


def bbox_gap_m(xy: tuple[float, float], bbox: dict) -> float:
    """Horizontal distance from `xy` to `bbox`; 0.0 if inside it.

    Same measure `nearest_obstacle_clearance_m` reports for the generated
    set, so the two can be compared and min()'d directly.
    """
    x, y = xy
    dx = max(bbox["x_min"] - x, 0.0, x - bbox["x_max"])
    dy = max(bbox["y_min"] - y, 0.0, y - bbox["y_max"])
    return math.hypot(dx, dy)


def point_clears_island(
    xy: tuple[float, float],
    half_width: float = BASE_HALF_WIDTH,
    bbox: dict = KITCHEN_ISLAND_BBOX,
) -> bool:
    """True if a base centred at `xy` (footprint `xy` +/- `half_width`)
    does not overlap the kitchen island bbox."""
    x, y = xy
    return not (
        bbox["x_min"] - half_width <= x <= bbox["x_max"] + half_width
        and bbox["y_min"] - half_width <= y <= bbox["y_max"] + half_width
    )


def route_avoiding_island(
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    """Waypoint route between two KITCHEN-SIDE points (both south of the
    door) that may straddle the island in x while both y's sit inside its
    span -- a straight `navigate_to(*target_xy)` between them drives the
    base directly into the island's real PhysX collider.

    2026-08-09 (O1 investigation): GPU-confirmed on `_push_object_to`'s own
    stance leg -- 3/3 runs (seed 42 x2, seed 7 x1, fully deterministic, no
    randomness engaged on this path) stalled with `ProgressWatchdog`
    firing (<1cm/5s) at the exact geometry this function detects: base
    starts east-clear of `KITCHEN_ISLAND_BBOX`, the chosen push stance
    sits west-clear of it, and the straight line between them crosses
    the island's real x-span at a y still inside its y-span.
    `route_via_door` already solves the equivalent problem for the
    door-crossing case (its own `ISLAND_CLEAR_X` detour, lines above) but
    only inside that function's cross-door branch; same-side kitchen
    moves (this case) fell through to a plain two-point route with no
    island check at all. Detours north to `TASK3_KITCHEN_LANE_Y` -- an
    already-proven-clear east-west lane (`route_via_door` uses it for the
    same purpose) -- rather than inventing new geometry.
    """
    sx, sy = start_xy
    tx, ty = target_xy
    x_lo = KITCHEN_ISLAND_BBOX["x_min"] - BASE_HALF_WIDTH
    x_hi = KITCHEN_ISLAND_BBOX["x_max"] + BASE_HALF_WIDTH
    y_lo = KITCHEN_ISLAND_BBOX["y_min"] - BASE_HALF_WIDTH
    y_hi = KITCHEN_ISLAND_BBOX["y_max"] + BASE_HALF_WIDTH
    straddles_x = (sx >= x_hi and tx <= x_lo) or (sx <= x_lo and tx >= x_hi)
    either_y_inside = (y_lo <= sy <= y_hi) or (y_lo <= ty <= y_hi)
    if not (straddles_x and either_y_inside):
        return [start_xy, target_xy]
    lane_y = TASK3_KITCHEN_LANE_Y
    return [start_xy, (sx, lane_y), (tx, lane_y), target_xy]


# T4 (plans/LOOP_PROMPT_VM_A_REV5.md): all four Stage-1/4 objects spawn
# INSIDE KITCHEN_ISLAND_BBOX (STATUS_2026-08-04.md sec 3), so any candidate
# stance close enough to reach one sits right at the island's edge -- a
# straight line from wherever the robot currently is to that candidate very
# often has to cut through the island's interior on the way (the island
# sits, geometrically, between "elsewhere in the kitchen" and "right next
# to an object on the counter"). A live GPU run (2026-08-04 07:40 UTC,
# plans/SYNC.md) confirmed this empirically: 9/9 curobo_stance_for() calls
# fell back, 100% for exactly this reason. `_segment_clears_island` below
# (moved here from curobo_stance.py -- this is where it always should have
# lived, and matches this file's stated ownership) is deliberately kept
# strict; `route_around_island` is the additive fix -- it tries a U-shaped
# detour over/under/around the island's convex footprint before giving up,
# instead of rejecting outright.
_PATH_CLEARANCE_SAMPLES = 20

# The sample COUNT above is a floor, not the whole story. Sampling a leg at a
# fixed count makes the check's spatial resolution depend on the leg's length,
# and that is not a rounding detail -- it inverts the result. Measured
# 2026-08-15 on the `navigate_rotate_spot` leg:
#
#   (-3.179,-3.1) -> (-3.300,-3.1)   len 0.121 m   20 samples @ 6 mm   -> BLOCKED
#   (-3.179,-3.1) -> (-4.960,-3.1)   len 1.781 m   20 samples @ 89 mm  -> CLEAR
#
# Both legs run along the same line y=-3.1 and the second CONTAINS the first,
# so "clear" is impossible: the coarse samples simply stepped over the one
# violation (real minimum clearance 0.3646 m against `Line042`, vs the 0.40 m
# half-width both legs are judged by). `route_around_island` then preferred
# that superset leg as a "detour", and `NavigateTo` drove the base 1.78 m west
# to (-4.96,-3.1) -- the exact x~-4.96 zone `_rotate_to_clear_island`'s own
# comment already flags as bad. That is what failed `navigate_rotate_spot`
# with the base parked at (-3.989,-3.067), 0.69 m off target, having used its
# whole 35 s budget: not a wheel-drive stall and not the object, but a route
# that asked for it. The same 55 historical runs reached (-3.294,-3.066).
#
# Sampling at a fixed SPACING makes the check a property of the geometry
# rather than of how the route happened to be split into waypoints. 0.02 m is
# an order of magnitude below `BASE_HALF_WIDTH`, so no obstacle wide enough to
# matter can hide between two samples. Denser sampling is strictly more
# conservative -- it can only reject legs the old check accepted, and a leg
# with no clearing detour is passed through unchanged (see
# `insert_island_detours`), which is the pre-2026-08-09 behaviour.
_PATH_CLEARANCE_SPACING_M = 0.02
ISLAND_DETOUR_MARGIN_M = 0.05

# Correction (plans/SYNC.md 2026-08-04 ~16:05 UTC): a live A/B test (25s vs
# 60s navigate_to budget, identical outcome either way) found the real
# navigate_to failure isn't a time shortage -- the base froze at the SAME
# position across 6 consecutive attempts, 0.31m from a real, already-
# GENERATED room obstacle (`assets/derived/room_obstacles.json`, part of
# what looks like the kitchen/dining partition wall) that `_segment_clears_
# island` never checked -- only `KITCHEN_ISLAND_BBOX`. `load_room_obstacles`/
# `nearest_obstacle_clearance_m` already existed in this file (added N5) but
# were never wired into any routing decision. Cached module-level (not
# re-read from disk on every one of ~20 samples per leg, of which there can
# be many per candidate during a stance search).
_ROOM_OBSTACLES_CACHE: list[dict] | None = None


def _cached_room_obstacles() -> list[dict]:
    global _ROOM_OBSTACLES_CACHE
    if _ROOM_OBSTACLES_CACHE is None:
        _ROOM_OBSTACLES_CACHE = load_room_obstacles()
    return _ROOM_OBSTACLES_CACHE


def _segment_clears_island(
    p0: tuple[float, float],
    p1: tuple[float, float],
    half_width: float = BASE_HALF_WIDTH,
    bbox: dict = KITCHEN_ISLAND_BBOX,
    n_samples: int = _PATH_CLEARANCE_SAMPLES,
    obstacles: list[dict] | None = None,
) -> bool:
    """True if the straight segment p0->p1 never overlaps the kitchen
    island OR any real generated room obstacle, sampled at `n_samples`
    interior points.

    NavigateTo (task3_autonomy/skills.py) does NOT drive a bare straight
    line -- it always routes via `route_via_door()` first (y-then-x or
    x-then-y waypoints) and, since this task, `insert_island_detours()`
    on top of that. This function is the pure per-LEG clearance primitive
    both `route_around_island` and the caller's own pre-flight candidate
    filter build on top of -- it intentionally still checks one straight
    leg at a time, not a whole multi-waypoint route. [GATE B0 follow-up,
    handoff sec 69: observed live, 3 repeated navigate_to stalls on the
    same cuRobo-chosen candidate, real run log -- the reason this exists
    at all and why it stays strict rather than being loosened.]

    `obstacles`: `None` (default, real production behavior) lazily loads
    and caches the real generated set. Callers that want ISLAND-ONLY
    semantics in isolation (this file's own existing tests, which use
    synthetic coordinates that were never validated against real room
    geometry) pass `obstacles=[]` explicitly -- `nearest_obstacle_
    clearance_m` returns `math.inf` for an empty list, a true no-op.
    """
    if obstacles is None:
        obstacles = _cached_room_obstacles()
    # `n_samples` is the floor; long legs get proportionally more samples so
    # the resolution stays fixed in metres. See `_PATH_CLEARANCE_SPACING_M`.
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n_samples = max(
        n_samples, int(math.ceil(length / _PATH_CLEARANCE_SPACING_M))
    )
    for i in range(1, n_samples):
        t = i / n_samples
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        if not point_clears_island((x, y), half_width, bbox):
            return False
        if nearest_obstacle_clearance_m((x, y), obstacles) < half_width:
            return False
    return True


def _island_detour_bounds(
    half_width: float = BASE_HALF_WIDTH,
    bbox: dict = KITCHEN_ISLAND_BBOX,
    margin: float = ISLAND_DETOUR_MARGIN_M,
) -> tuple[float, float, float, float]:
    return (
        bbox["x_min"] - half_width - margin,
        bbox["x_max"] + half_width + margin,
        bbox["y_min"] - half_width - margin,
        bbox["y_max"] + half_width + margin,
    )


def route_around_island(
    p0: tuple[float, float],
    p1: tuple[float, float],
    half_width: float = BASE_HALF_WIDTH,
    bbox: dict = KITCHEN_ISLAND_BBOX,
    obstacles: list[dict] | None = None,
) -> list[tuple[float, float]] | None:
    """A route from `p0` to `p1` that clears the kitchen island via a
    single U-shaped detour (over the top, under the bottom, or around
    either side of the expanded bbox), or `None` if none of the 4 works.

    Returns `[p0, p1]` unchanged when the straight leg already clears --
    zero behavior change for the (still-common) case that dominated this
    function's callers before today. A single BBOX-CORNER waypoint (tried
    first, in an earlier version of this function) only routes around the
    island when `p0` and `p1` are already on the same side of it in one
    axis -- it cannot connect two points straddling OPPOSITE sides (e.g.
    robot east of the island, candidate west of it, both near the same
    y-band -- exactly the case the 2026-08-04 07:40 UTC live episode hit
    9/9 times, plans/SYNC.md). Going over/under/around one full side
    (`p0 -> (p0.x_or_y, edge) -> (p1.x_or_y, edge) -> p1`) handles that
    case; a degenerate leg (start and end already share that coordinate)
    collapses to a 2-point route automatically. Purely additive: this only
    ever ACCEPTS a route the old straight-only check rejected, never the
    reverse -- every leg of every candidate route is verified by the same
    strict per-leg check that already governed the old behavior, so this
    cannot reintroduce the stall it exists to prevent (handoff sec 69).
    """
    if obstacles is None:
        obstacles = _cached_room_obstacles()
    if _segment_clears_island(p0, p1, half_width, bbox, obstacles=obstacles):
        return [p0, p1]
    x_min, x_max, y_min, y_max = _island_detour_bounds(half_width, bbox)
    candidate_routes = [
        [p0, (p0[0], y_max), (p1[0], y_max), p1],  # over the north side
        [p0, (p0[0], y_min), (p1[0], y_min), p1],  # under the south side
        [p0, (x_max, p0[1]), (x_max, p1[1]), p1],  # around the east side
        [p0, (x_min, p0[1]), (x_min, p1[1]), p1],  # around the west side
    ]
    best_length = None
    best_route = None
    for route in candidate_routes:
        pts = [route[0]]
        for pt in route[1:]:
            if pt != pts[-1]:  # drop degenerate zero-length legs
                pts.append(pt)
        if len(pts) < 2:
            continue
        if not all(
            _segment_clears_island(a, b, half_width, bbox, obstacles=obstacles)
            for a, b in zip(pts, pts[1:])
        ):
            continue
        length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])
        )
        if best_length is None or length < best_length:
            best_length = length
            best_route = pts
    return best_route


def insert_island_detours(
    waypoints: list[tuple[float, float]],
    obstacles: list[dict] | None = None,
) -> list[tuple[float, float]]:
    """Post-process a waypoint route (e.g. `route_via_door`'s output),
    inserting `route_around_island`'s detour waypoints between any
    consecutive pair whose straight leg would cross the kitchen island
    or a real generated room obstacle.

    Additive and order-preserving: a route with no crossing leg comes
    back byte-for-byte identical. A leg `route_around_island` can't clear
    via any detour is passed through unchanged too (this function only
    ever adds waypoints, never drops the original ones) -- the caller's
    own existing stall protections still apply to it.

    `obstacles`: see `_segment_clears_island` -- `None` (default) is the
    real production behavior; pass `[]` for island-only test isolation.
    """
    if len(waypoints) < 2:
        return list(waypoints)
    result = [waypoints[0]]
    for i in range(1, len(waypoints)):
        p0, p1 = waypoints[i - 1], waypoints[i]
        detour = route_around_island(p0, p1, obstacles=obstacles)
        if detour is not None:
            result.extend(detour[1:-1])  # intermediate waypoints only
        result.append(p1)
    return result


def _rotate_to_clear_island(
    object_xy: tuple[float, float],
    nominal_angle_rad: float,
    radius: float,
    *,
    half_width: float = BASE_HALF_WIDTH,
    step_rad: float = math.radians(2.0),
    max_sweep_rad: float = math.pi,
    margin_m: float = ISLAND_DETOUR_MARGIN_M,
    max_radius_m: float | None = None,
    radius_step_m: float = 0.02,
) -> tuple[tuple[float, float], float]:
    """Slide a stance candidate around `object_xy` at fixed `radius`,
    starting at `nominal_angle_rad` and sweeping outward (alternating
    +/-) until the footprint clears the island AND every other real
    generated room obstacle. Preserves `radius` exactly (the reach
    budget), only the direction changes -- ACTIVE_BRIEF.md sec 3.2/T2: a
    candidate calibrated for one object position can end up inside the
    island once the object's live pose drifts; sliding around the object
    at the same reach distance finds the nearest clear direction instead
    of hardcoding a per-object correction.

    2026-08-09 (O1 investigation): this function only ever checked
    `point_clears_island` (the hand-typed `KITCHEN_ISLAND_BBOX`), so a
    swept candidate could clear the island while landing inside a
    DIFFERENT real collider -- GPU-confirmed 2x (`cup_run4_islandfix.log`,
    `spoon2_run2_navfix.log`/`spoon2_run3_nocurobo.log`, both curobo AND
    this fallback): the swept "east"/"north" stance for objects near the
    west counter repeatedly landed at x~-4.86 to -4.96, inside
    `/root/root/Line041/Line041` (the room's real west wall,
    `x=[-5.947,-4.067]`), and the base then sat physically stuck there for
    the rest of the episode. `_segment_clears_island` (below) already
    checks `nearest_obstacle_clearance_m` against the SAME generated
    `load_room_obstacles()` set this function never consulted -- this was
    a real gap between two sibling functions, not a missing feature.

    NOTE on the "x~-4.86 to -4.96 is inside the west wall" claim above: it
    was half right. Those candidates ARE bad, but not because of a wall --
    `WEST_WALL_BBOX`'s old value was wrong by 1.64 m (see its comment).
    x=-4.86..-4.96 sits at `KITCHEN_ISLAND_BBOX`'s own inflated west edge
    (x_min - half_width = -4.91) with millimetre-scale real clearance,
    which is the same margin-less-boundary defect `curobo_stance.py`
    already fixed on its own candidate grid with `ISLAND_DETOUR_MARGIN_M`.
    This function now requires that same margin instead of a phantom wall.

    Returns `(xy, angle_rad)` -- the angle so the caller can face the base
    at the object rather than at the nominal direction it asked for.

    `max_radius_m` -- how far the radius may GROW when no angle at all
    clears at the requested one. Growth exists because a requested radius
    can be geometrically impossible rather than merely awkward: the Stage-4
    push radius (`world_isaac.PUSH_STANCE_RADIUS_M`, 0.48 m) puts the base
    centre 0.48 m from an object that itself sits only ~0.20 m in from the
    counter's west edge, so the ENTIRE annulus at that radius lies inside
    the counter's inflated footprint -- 0 of 180 angles clear, for every
    Stage-1/4 kitchen object. Growing to ~0.65 m is the first radius at
    which any stance is legal for that caller.

    **`None` (the default) means NO growth: the requested radius is the
    ceiling.** It used to default to `MEASURED_REACH_LIMIT_M`, which made
    growth implicit for every caller including the grasp path -- and the
    grasp path cannot afford it. `STANCE_REACH_RADIUS_M` is not a
    preference there: `reach()`'s own grasp calibration is measured at
    exactly that distance, and the constant was scaled down to ~0.78 m
    specifically to hold margin against the ~0.855 m measured kinematic
    ceiling (see its own comment). Measured, spoon2 at (-4.1, -1.7): the
    best candidate at the requested radius fell 3 mm short of the 0.05 m
    obstacle margin, so this function pushed the stance to 0.79997 m --
    20 mm past the reach budget -- and nothing in the return value or the
    logs said so. Trading a 3 mm margin shortfall for a 20 mm reach-budget
    overrun is the wrong trade, and making it silently is worse than
    making it wrong. A caller that genuinely needs growth now asks for it
    by name, and any growth that does happen is logged.
    """
    ceiling = radius if max_radius_m is None else max(radius, max_radius_m)

    def candidate(angle: float, r: float) -> tuple[float, float]:
        return (
            object_xy[0] + r * math.cos(angle),
            object_xy[1] + r * math.sin(angle),
        )

    def margin(xy: tuple[float, float]) -> float:
        """Clearance BEYOND the base footprint. <= 0 means overlapping."""
        return (
            min(
                bbox_gap_m(xy, KITCHEN_ISLAND_BBOX),
                bbox_gap_m(xy, WEST_WALL_BBOX),
                nearest_obstacle_clearance_m(xy, _cached_room_obstacles()),
            )
            - half_width
        )

    best_xy = candidate(nominal_angle_rad, radius)
    best_angle = nominal_angle_rad
    best_margin = margin(best_xy)

    r = radius
    while True:
        xy = candidate(nominal_angle_rad, r)
        xy_margin = margin(xy)
        if xy_margin >= margin_m:
            return xy, nominal_angle_rad
        if xy_margin > best_margin:
            best_margin, best_xy, best_angle = xy_margin, xy, nominal_angle_rad
        delta = step_rad
        while delta <= max_sweep_rad:
            for sign in (1.0, -1.0):
                angle = nominal_angle_rad + sign * delta
                xy = candidate(angle, r)
                xy_margin = margin(xy)
                if xy_margin >= margin_m:
                    return xy, angle
                if xy_margin > best_margin:
                    best_margin, best_xy, best_angle = xy_margin, xy, angle
            delta += step_rad
        if r >= ceiling:
            break
        r = min(r + radius_step_m, ceiling)
        # Never silently. A grown radius means the base ends up farther
        # from the object than the caller's reach budget allowed for, and
        # that is exactly the condition under which reach() solves IK
        # against a target it cannot physically get to.
        print(
            "STANCE_RADIUS_GROWN "
            f"object_xy=({object_xy[0]:.3f},{object_xy[1]:.3f}) "
            f"requested_m={radius:.4f} now_m={r:.4f} ceiling_m={ceiling:.4f} "
            f"required_margin_m={margin_m:.3f} best_margin_m={best_margin:.4f}",
            flush=True,
        )
    # Swept every angle at every radius up to the reach limit and nothing
    # cleared. Returning the NOMINAL
    # here (the behaviour until 2026-08-14) is the worst available answer:
    # the nominal is the one candidate already known to be unchecked, and
    # for every kitchen-counter object it is a point INSIDE the counter's
    # own inflated footprint. The base then drives into the counter and
    # jams -- which is precisely the "base-drive dead zone at x~-4.3 to
    # -4.5" that three sessions attributed to the wheel drive. Return the
    # least-overlapping candidate instead, and say so loudly: a silent
    # geometric fallback is how this survived that long.
    print(
        "STANCE_NO_CLEAR_CANDIDATE "
        f"object_xy=({object_xy[0]:.3f},{object_xy[1]:.3f}) "
        f"radius={radius:.3f}..{ceiling:.3f} "
        f"half_width={half_width:.3f} "
        f"required_margin_m={margin_m:.3f} best_margin_m={best_margin:.3f} "
        f"best_xy=({best_xy[0]:.3f},{best_xy[1]:.3f}) "
        f"best_angle_deg={math.degrees(best_angle):.1f}",
        flush=True,
    )
    return best_xy, best_angle


def stance_for(
    object_xy: tuple[float, float],
    approach: str,
    *,
    radius_m: float | None = None,
    max_radius_m: float | None = None,
) -> tuple[tuple[float, float], float]:
    """Reach-safe base stance for `object_xy`, clamped clear of the kitchen
    island footprint (see `_rotate_to_clear_island`). `approach` selects the
    nominal direction ("east" or "north" fallback); the reach radius
    (`STANCE_REACH_RADIUS_M` by default) is preserved, only the direction
    changes.

    `radius_m` (R7 T2, plans/SYNC.md 2026-08-04 ~19:48 UTC): `None` (default)
    is the original, unaffected `STANCE_REACH_RADIUS_M` -- this project's
    grasp `reach()` path calibrates against this exact radius and must never
    see a different one. Push's own stance call passes an explicit override:
    R7 T1's IK feasibility sweep found the CURRENT default radius sits
    exactly in a 0%-feasible zone for push_approach (0/60,750 sampled combos
    at this stance-to-object distance or farther), while moving the stance
    ~0.3-0.4m closer to the object raised feasibility to 6.8-16.8% -- a real,
    measured lever, not a guessed constant.

    `max_radius_m`: `None` (default) means the stance is never placed
    farther from the object than `radius`. Pass an explicit ceiling only
    from a caller that can tolerate a longer reach -- see
    `_rotate_to_clear_island` for why this is opt-in rather than assumed.
    """
    radius = STANCE_REACH_RADIUS_M if radius_m is None else radius_m
    if approach == "east":
        nominal_angle = math.atan2(*reversed(STANCE_OFFSET_EAST))
        yaw = STANCE_YAW_EAST_RAD
    else:
        nominal_angle = math.pi / 2.0  # due north
        yaw = -math.pi / 2.0
    xy, angle = _rotate_to_clear_island(
        object_xy, nominal_angle, radius, max_radius_m=max_radius_m
    )
    # 2026-08-14: the yaw used to be pinned to `approach` no matter how far
    # the sweep had to rotate the stance, so a stance swept 90 deg round
    # the object still faced the ORIGINAL direction -- the object ended up
    # beside or behind the robot, and the arm solved IK against thin air.
    # (Real signature: `ik_ok_ticks: 0/1200`, `position_error_m ~1.6` on
    # 2026-08-14's attempts 2 and 3, with the base parked at the stance.)
    # The nominal yaws above are both exactly "face the object from the
    # nominal angle", so deriving yaw from the angle actually used is the
    # same rule, applied consistently. Left byte-identical when the sweep
    # did not move: only the rotated case changes.
    if angle != nominal_angle:
        yaw = wrap_to_pi(angle + math.pi)
    return xy, yaw


# N5 (plans/OVERNIGHT_VM_B_2026-08-03.md): KITCHEN_ISLAND_BBOX above is one
# hand-typed box for a room `probe_room_geometry_near_point.py` found 31
# separate geometry hits near a single point in. This is the additive fix:
# a GENERATED set of real bboxes from the room USD itself
# (scripts/task3/generate_room_obstacles.py, regeneration command in its
# own header), consulted alongside -- not instead of -- the hand-typed
# constant. Pure-JSON, no Isaac import, so this stays CPU-testable per
# this file's own rule.
_ROOM_OBSTACLES_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "derived"
    / "room_obstacles.json"
)


def load_room_obstacles(path: Path = _ROOM_OBSTACLES_PATH) -> list[dict]:
    """Load the generated obstacle set. Returns `[]` (not an error) if the
    file hasn't been generated yet -- callers get pure hand-typed-bbox
    behavior in that case, matching this feature's additive contract."""
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("obstacles", [])


def obstacle_blocks_base(ob: dict) -> bool:
    """True if `ob`'s z-span overlaps the band the robot's body occupies.

    A box with no z recorded is treated as blocking (fail safe). See
    `BASE_BLOCKING_Z_MIN_M` / `BASE_BLOCKING_Z_MAX_M` for why the band is
    where it is.
    """
    z_min = ob.get("z_min")
    z_max = ob.get("z_max")
    if z_min is None or z_max is None:
        return True
    return z_max > BASE_BLOCKING_Z_MIN_M and z_min < BASE_BLOCKING_Z_MAX_M


def nearest_obstacle_clearance_m(
    xy: tuple[float, float],
    obstacles: list[dict] | None = None,
    *,
    z_filter: bool = True,
) -> float:
    """Real clearance (metres) from `xy` to the nearest generated-obstacle
    bbox, or `math.inf` if the obstacle set is empty (nothing generated
    yet, or the point is genuinely far from everything). 0.0 means `xy` is
    INSIDE an obstacle's footprint -- a real, hard violation, not close-to
    but clear.

    `z_filter` (default on, 2026-08-14): skip boxes that cannot touch the
    robot's body at all -- floor decals below the wheel contact plane and
    overhead fixtures above the robot's height. Pass `z_filter=False` for
    the old purely-2D behaviour.
    """
    if obstacles is None:
        obstacles = load_room_obstacles()
    if not obstacles:
        return math.inf
    x, y = xy
    best = math.inf
    for ob in obstacles:
        if z_filter and not obstacle_blocks_base(ob):
            continue
        dx = max(ob["x_min"] - x, 0.0, x - ob["x_max"])
        dy = max(ob["y_min"] - y, 0.0, y - ob["y_max"])
        dist = math.hypot(dx, dy)
        if dist < best:
            best = dist
    return best
