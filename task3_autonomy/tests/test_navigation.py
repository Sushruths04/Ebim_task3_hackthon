# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task3_autonomy.navigation import (  # noqa: E402
    BASE_HALF_WIDTH,
    ISLAND_DETOUR_MARGIN_M,
    KITCHEN_ISLAND_BBOX,
    STANCE_REACH_RADIUS_M,
    TASK3_DOOR_APPROACH_M,
    TASK3_DOOR_X,
    TASK3_DOOR_Y,
    TASK3_KITCHEN_LANE_Y,
    Pose2D,
    ProgressWatchdog,
    WEST_WALL_BBOX,
    _segment_clears_island,
    base_twist_toward,
    bbox_gap_m,
    insert_island_detours,
    load_room_obstacles,
    nearest_obstacle_clearance_m,
    point_clears_island,
    pose_reached,
    route_around_island,
    route_via_door,
    stance_for,
    waypoints_x_then_y,
    waypoints_y_then_x,
    wrap_to_pi,
)

pytest_approx = pytest.approx

# S1's own live GPU finding (GATE S1 CONFIRMED, s1_gpu_confirm_r2): the
# ~0.855m arm reach limit, measured, not a spec value invented for this test.
_MEASURED_REACH_LIMIT_M = 0.855


def test_wrap_to_pi_keeps_values_already_in_range():
    assert wrap_to_pi(0.5) == pytest_approx(0.5)


def test_wrap_to_pi_wraps_values_outside_range():
    # +-pi is the wrap boundary itself (both signs are the same angle), so
    # avoid asserting a sign exactly at the boundary -- use interior points.
    assert wrap_to_pi(2.5 * math.pi) == pytest_approx(0.5 * math.pi)
    assert wrap_to_pi(-2.5 * math.pi) == pytest_approx(-0.5 * math.pi)
    assert abs(wrap_to_pi(3.0 * math.pi)) == pytest_approx(math.pi)
    assert abs(wrap_to_pi(-3.0 * math.pi)) == pytest_approx(math.pi)


def test_waypoints_y_then_x_routes_through_y_first_when_both_change():
    waypoints = waypoints_y_then_x((0.0, 0.0), (3.0, 4.0))
    assert waypoints == [(0.0, 0.0), (0.0, 4.0), (3.0, 4.0)]


def test_waypoints_y_then_x_collapses_when_x_already_aligned():
    waypoints = waypoints_y_then_x((0.0, 0.0), (0.0, 4.0))
    assert waypoints == [(0.0, 0.0), (0.0, 4.0)]


def test_waypoints_y_then_x_collapses_when_y_already_aligned():
    waypoints = waypoints_y_then_x((0.0, 0.0), (3.0, 0.0))
    assert waypoints == [(0.0, 0.0), (3.0, 0.0)]


def test_waypoints_y_then_x_is_single_point_when_already_at_target():
    waypoints = waypoints_y_then_x((1.0, 1.0), (1.0, 1.0))
    assert waypoints == [(1.0, 1.0)]


def test_waypoints_x_then_y_routes_through_x_first_when_both_change():
    waypoints = waypoints_x_then_y((0.0, 0.0), (3.0, 4.0))
    assert waypoints == [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]


def test_waypoints_x_then_y_is_single_point_when_already_at_target():
    waypoints = waypoints_x_then_y((1.0, 1.0), (1.0, 1.0))
    assert waypoints == [(1.0, 1.0)]


def test_route_via_door_same_side_falls_back_to_y_then_x():
    start, target = (-4.6, 2.7), (-1.0, 1.5)  # both north of the partition
    assert route_via_door(start, target) == waypoints_y_then_x(start, target)


NORTH_POINT = (TASK3_DOOR_X, TASK3_DOOR_Y + TASK3_DOOR_APPROACH_M)
SOUTH_POINT = (TASK3_DOOR_X, TASK3_KITCHEN_LANE_Y)


def test_route_via_door_crossing_passes_through_doorway_center():
    route = route_via_door((-4.6, 2.7), (-3.18, -1.6))
    # The two door waypoints must be consecutive: the crossing leg is a
    # straight line at the doorway's x, never a diagonal near the wall.
    approach_i = route.index(NORTH_POINT)
    assert route[approach_i + 1] == SOUTH_POINT
    assert route[0] == (-4.6, 2.7)
    assert route[-1] == (-3.18, -1.6)
    # The kitchen-side east leg runs in the shallow lane: after the door,
    # y stays at the lane until the final x is reached.
    assert (route[approach_i + 2][1]) == TASK3_KITCHEN_LANE_Y


def test_route_via_door_crossing_south_to_north_is_mirrored():
    route = route_via_door((-3.18, -1.6), (-4.6, 2.7))
    approach_i = route.index(SOUTH_POINT)
    assert route[approach_i + 1] == NORTH_POINT
    assert route[-1] == (-4.6, 2.7)


# T2 (ACTIVE_BRIEF.md sec 3.2/4): object x positions spanning the observed
# live drift (spawn -4.185 to the furthest live read -4.31), each with two
# real observed y values, so stance_for() is exercised against the exact
# geometry that stalled real GPU runs.
_DRIFT_OBJECT_XY = [
    (-4.185, -1.753),
    (-4.20, -1.70),
    (-4.25, -1.68),
    (-4.31, -1.6609),
]


@pytest.mark.parametrize("object_xy", _DRIFT_OBJECT_XY)
@pytest.mark.parametrize("approach", ["east", "north"])
def test_stance_for_clears_island_across_observed_drift(object_xy, approach):
    stance_xy, _yaw = stance_for(object_xy, approach)
    assert point_clears_island(stance_xy), (approach, object_xy, stance_xy)


@pytest.mark.parametrize("object_xy", _DRIFT_OBJECT_XY)
@pytest.mark.parametrize("approach", ["east", "north"])
def test_stance_for_preserves_reach_budget(object_xy, approach):
    stance_xy, _yaw = stance_for(object_xy, approach)
    dist = math.hypot(stance_xy[0] - object_xy[0], stance_xy[1] - object_xy[1])
    assert dist <= STANCE_REACH_RADIUS_M + 1e-9
    assert dist <= 0.87


# R7 T2 (plans/SYNC.md 2026-08-04 ~19:48 UTC): the offline IK feasibility
# sweep's dominant finding -- push_approach needs a stance closer to the
# object than STANCE_REACH_RADIUS_M provides. `radius_m` is opt-in
# (default None) so `reach()`'s own grasp calibration, which depends on
# the exact default radius, is provably unaffected.
def test_stance_for_default_radius_unchanged_when_override_omitted():
    object_xy = (-4.185, -1.753)
    with_default, yaw_default = stance_for(object_xy, "east")
    with_none, yaw_none = stance_for(object_xy, "east", radius_m=None)
    assert with_default == with_none
    assert yaw_default == yaw_none


@pytest.mark.parametrize("object_xy", _DRIFT_OBJECT_XY)
def test_stance_for_radius_override_is_honored(object_xy):
    """The override is a REQUEST and a floor, not a hard cap.

    Changed 2026-08-14. The assertion used to be `dist <= override_radius`,
    i.e. the radius may never grow. Real geometry says that cap cannot
    always be met: these objects sit ~0.20 m in from the kitchen counter's
    west edge, so at the push override radius (0.48 m) the base centre is
    0.28 m from the counter -- inside its own 0.40 m footprint at EVERY
    angle, 0 of 180. The old code satisfied this assertion by returning an
    overlapping stance, and the base then drove into the counter and
    jammed. `_rotate_to_clear_island` now grows the radius, and only when
    nothing clears, to at most the measured reach ceiling.
    """
    override_radius = STANCE_REACH_RADIUS_M - 0.30
    stance_xy, _yaw = stance_for(object_xy, "east", radius_m=override_radius)
    dist = math.hypot(stance_xy[0] - object_xy[0], stance_xy[1] - object_xy[1])
    assert dist >= override_radius - 1e-9, "radius must never SHRINK"
    assert dist <= _MEASURED_REACH_LIMIT_M + 1e-9, (
        "a grown radius must still be reachable"
    )
    # The override must never leave the base FARTHER out than the default
    # radius would have. It cannot be asserted to land strictly closer:
    # for an object whose own geometry puts the first legal radius at or
    # above the default (e.g. one sitting deep behind the counter edge),
    # "as close as legal" and "the default" are the same point, and that
    # is the right answer rather than a silent fallback.
    default_xy, _ = stance_for(object_xy, "east")
    default_dist = math.hypot(
        default_xy[0] - object_xy[0], default_xy[1] - object_xy[1]
    )
    assert dist <= default_dist + 1e-9


@pytest.mark.parametrize("object_xy", _DRIFT_OBJECT_XY)
@pytest.mark.parametrize("approach", ["east", "north"])
# (radius_m, max_radius_m) as the two REAL call sites pass them. The grasp
# path takes the default radius with no growth ceiling -- its reach budget
# is what `reach()` is calibrated against and may not be exceeded. The push
# path overrides the radius down to 0.48 m, where the entire annulus lies
# inside the counter, and must therefore name a growth ceiling explicitly;
# before 2026-08-21 that ceiling was the silent default for both callers,
# which let a grasp stance drift 20 mm past its own budget unannounced.
@pytest.mark.parametrize(
    "radius_m,max_radius_m",
    [(None, None), (STANCE_REACH_RADIUS_M - 0.30, _MEASURED_REACH_LIMIT_M)],
)
def test_stance_for_never_returns_a_stance_inside_an_obstacle(
    object_xy, approach, radius_m, max_radius_m
):
    """The regression guard for the 2026-08-14 root cause.

    `_rotate_to_clear_island` used to return the UNCHECKED nominal
    candidate whenever its sweep found nothing, and with `WEST_WALL_BBOX`
    wrong by 1.64 m the sweep found nothing for every kitchen object. The
    returned stance sat inside the counter's own inflated footprint; six
    real GPU episodes drove into the counter and were written up as a
    'base-drive dead zone'. A stance that fails its own clearance test
    must never be returned silently again.
    """
    stance_xy, _yaw = stance_for(
        object_xy, approach, radius_m=radius_m, max_radius_m=max_radius_m
    )
    gap = min(
        bbox_gap_m(stance_xy, KITCHEN_ISLAND_BBOX),
        bbox_gap_m(stance_xy, WEST_WALL_BBOX),
        nearest_obstacle_clearance_m(stance_xy),
    )
    assert gap - BASE_HALF_WIDTH >= ISLAND_DETOUR_MARGIN_M - 1e-9, (
        f"{approach} stance {stance_xy} for object {object_xy} clears by "
        f"only {gap - BASE_HALF_WIDTH:.4f} m"
    )


@pytest.mark.parametrize("object_xy", _DRIFT_OBJECT_XY)
@pytest.mark.parametrize("approach", ["east", "north"])
def test_stance_for_faces_the_object_it_was_swept_to(object_xy, approach):
    """A swept stance must face the object, not the nominal direction.

    The yaw used to be pinned to `approach` however far the sweep had
    rotated the stance, so the base could park correctly and still have
    the object beside or behind it -- the real 2026-08-14 signature was
    `ik_ok_ticks: 0/1200` with `position_error_m ~1.6` at a stance the
    base had actually reached.
    """
    stance_xy, yaw = stance_for(object_xy, approach)
    bearing = math.atan2(
        object_xy[1] - stance_xy[1], object_xy[0] - stance_xy[0]
    )
    assert abs(wrap_to_pi(bearing - yaw)) < math.radians(1.0)


# T4 (plans/LOOP_PROMPT_VM_A_REV5.md): shaped after a real robot
# position/target pair from the 2026-08-04 07:40 UTC live episode
# (plans/SYNC.md) that was observed to fall back -- "found NO
# practical-ok, path-clear candidate out of 18 searched (18 rejected for
# an island-crossing path from (-3.394..., -1.691...)". That exact x
# (-3.394) turned out to itself already fail `point_clears_island` at
# BASE_HALF_WIDTH margin (2.4cm past the threshold) -- moved to -3.0 so
# this test isolates the detour logic instead of that separate, real edge
# case (worth its own look: a live position landing just inside the
# margin this same check enforces elsewhere). Robot east of the island,
# candidate west of it, both near the island's y-band -- the straight leg
# is unclearable by construction (must cross the island's x-span at a y
# inside its range), so this is exactly the case route_around_island
# exists for.
_T4_ROBOT_EAST_OF_ISLAND = (-3.0, -1.691)
_T4_CANDIDATE_WEST_OF_ISLAND = (-4.93, -1.70)


def _straight_leg_samples(p0, p1, n=40):
    for i in range(n + 1):
        t = i / n
        yield (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)


def test_route_around_island_finds_a_detour_the_straight_line_cannot():
    assert not all(
        point_clears_island(pt)
        for pt in _straight_leg_samples(
            _T4_ROBOT_EAST_OF_ISLAND, _T4_CANDIDATE_WEST_OF_ISLAND
        )
    ), "test setup: this straight leg was supposed to cross the island"

    route = route_around_island(
        _T4_ROBOT_EAST_OF_ISLAND, _T4_CANDIDATE_WEST_OF_ISLAND, obstacles=[]
    )

    assert route is not None
    assert route[0] == _T4_ROBOT_EAST_OF_ISLAND
    assert route[-1] == _T4_CANDIDATE_WEST_OF_ISLAND
    for leg_start, leg_end in zip(route, route[1:]):
        assert all(
            point_clears_island(pt)
            for pt in _straight_leg_samples(leg_start, leg_end)
        ), (leg_start, leg_end)


def test_route_around_island_returns_straight_line_unchanged_when_clear():
    p0, p1 = (10.0, 10.0), (11.0, 11.0)  # far from KITCHEN_ISLAND_BBOX
    assert route_around_island(p0, p1, obstacles=[]) == [p0, p1]


def test_route_around_island_returns_none_when_target_itself_is_unreachable():
    # Target sits inside the island's own footprint -- no detour can help
    # a point_clears_island()-failing endpoint; this must not silently
    # "succeed" by returning a route ending inside furniture.
    inside_island = (
        (KITCHEN_ISLAND_BBOX["x_min"] + KITCHEN_ISLAND_BBOX["x_max"]) / 2.0,
        (KITCHEN_ISLAND_BBOX["y_min"] + KITCHEN_ISLAND_BBOX["y_max"]) / 2.0,
    )
    assert (
        route_around_island(
            _T4_ROBOT_EAST_OF_ISLAND, inside_island, obstacles=[]
        )
        is None
    )


def test_insert_island_detours_is_additive_when_no_leg_crosses():
    waypoints = [(10.0, 10.0), (10.0, 11.0), (11.0, 11.0)]
    assert insert_island_detours(waypoints, obstacles=[]) == waypoints


def test_insert_island_detours_fixes_a_same_side_kitchen_leg():
    # Mirrors what route_via_door's same-side branch (waypoints_y_then_x)
    # actually hands NavigateTo for a kitchen-local move like this one.
    raw = waypoints_y_then_x(
        _T4_ROBOT_EAST_OF_ISLAND, _T4_CANDIDATE_WEST_OF_ISLAND
    )
    routed = insert_island_detours(raw, obstacles=[])
    assert routed[0] == raw[0]
    assert routed[-1] == raw[-1]
    assert len(routed) > len(raw), "expected a detour waypoint to be inserted"


def test_insert_island_detours_also_checks_real_obstacles_by_default():
    # Correction (plans/SYNC.md 2026-08-04 ~16:05 UTC): a real generated
    # obstacle (kitchen/dining partition wall) sits ~0.31m from
    # (-2.0, -0.2) -- close enough that a naive island-only detour through
    # that area is unsafe. With the DEFAULT (obstacles=None, real cached
    # set), a route through that exact area must not be accepted as clear.
    near_real_wall = (-2.0, -0.08)
    assert nearest_obstacle_clearance_m(near_real_wall) < BASE_HALF_WIDTH, (
        "test setup: this point was supposed to be close to a real "
        "obstacle -- if this fails, room_obstacles.json content changed "
        "and this test's coordinates need updating, not deletion"
    )
    assert not _segment_clears_island((-2.0, -2.0), (-2.0, 1.0))


def test_clearance_check_resolution_does_not_depend_on_leg_length():
    """A leg that CONTAINS a blocked leg cannot itself be clear.

    Measured 2026-08-15 on the real `navigate_rotate_spot` leg: with a
    fixed 20-sample count, the 0.121 m direct leg was sampled every 6 mm
    and correctly rejected, while the 1.781 m leg along the SAME line
    y=-3.1 was sampled every 89 mm, stepped over the single violation
    (0.3646 m of real clearance against `Line042`, against a 0.40 m
    half-width) and was accepted. `route_around_island` then preferred
    that superset leg as its "detour" and `NavigateTo` drove the base
    1.78 m west, failing the phase 0.69 m off target.
    """
    blocked = ((-3.179, -3.1), (-3.3, -3.1))
    superset = ((-3.179, -3.1), (-4.96, -3.1))
    assert not _segment_clears_island(*blocked), (
        "test setup: this leg was supposed to be blocked by a real "
        "generated obstacle -- if this fails, room_obstacles.json content "
        "changed and this test's coordinates need updating, not deletion"
    )
    assert not _segment_clears_island(*superset)


def test_rotate_spot_leg_takes_no_westward_detour():
    """The proven corridor-stop -> rotate-spot route must stay a straight
    y-then-x run. 55 historical runs ended at (-3.294, -3.066); the run
    that took the spurious (-4.96, -3.1) detour ended at (-3.989, -3.067).
    """
    corridor_stop = (-3.179, -1.572)
    rotate_spot = (-3.3, -3.1)
    raw = route_via_door(corridor_stop, rotate_spot)
    assert insert_island_detours(raw) == raw


def test_S2_subfix3_reach_radius_leaves_real_margin_against_reach_ceiling():
    """S2 sub-fix 3 (P4): stance_for()'s fallback radius -- used whenever
    curobo_stance_for() finds no valid candidate at all -- must itself sit
    inside the measured ~0.855m reach ceiling. The old ~0.866m value made
    that fallback a guaranteed-unreachable stance by construction whenever
    it fired (GATE C2.5A)."""
    assert STANCE_REACH_RADIUS_M < _MEASURED_REACH_LIMIT_M
    margin_m = _MEASURED_REACH_LIMIT_M - STANCE_REACH_RADIUS_M
    assert margin_m >= 0.05, (
        f"STANCE_REACH_RADIUS_M={STANCE_REACH_RADIUS_M} leaves only "
        f"{margin_m:.3f}m of margin against the measured "
        f"{_MEASURED_REACH_LIMIT_M}m reach ceiling"
    )


def test_base_twist_toward_drives_straight_forward_in_body_frame():
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    vx, vy = base_twist_toward(
        pose, (1.0, 0.0), max_linear_mps=0.5, position_kp=1.5
    )
    assert vx == pytest_approx(0.5)
    assert vy == pytest_approx(0.0, abs=1e-9)


def test_base_twist_toward_drives_sideways_in_body_frame():
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    vx, vy = base_twist_toward(
        pose, (0.0, 1.0), max_linear_mps=0.5, position_kp=1.5
    )
    assert vx == pytest_approx(0.0, abs=1e-9)
    assert vy == pytest_approx(0.5)


def test_base_twist_toward_rotates_world_error_into_body_frame():
    # Facing +90 deg (world +y): a target ahead in world +x is to the
    # robot's right, i.e. negative body-frame y.
    pose = Pose2D(x=0.0, y=0.0, yaw=math.pi / 2.0)
    vx, vy = base_twist_toward(
        pose, (1.0, 0.0), max_linear_mps=0.5, position_kp=1.5
    )
    assert vx == pytest_approx(0.0, abs=1e-9)
    assert vy == pytest_approx(-0.5)


def test_base_twist_toward_is_proportional_when_close():
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    vx, vy = base_twist_toward(
        pose, (0.1, 0.0), max_linear_mps=0.5, position_kp=1.5
    )
    assert vx == pytest_approx(0.15)  # kp * distance, below the cap
    assert vy == pytest_approx(0.0, abs=1e-9)


def test_base_twist_toward_min_creep_floors_speed_when_close():
    # At distance=0.05, kp*distance=0.075 -- below a 0.18 floor.
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    vx, vy = base_twist_toward(
        pose,
        (0.05, 0.0),
        max_linear_mps=0.5,
        position_kp=1.5,
        min_creep_mps=0.18,
    )
    assert vx == pytest_approx(0.18)
    assert vy == pytest_approx(0.0, abs=1e-9)


def test_base_twist_toward_min_creep_does_not_override_faster_p_term():
    # At distance=0.2, kp*distance=0.3 -- already above a 0.18 floor.
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    vx, vy = base_twist_toward(
        pose,
        (0.2, 0.0),
        max_linear_mps=0.5,
        position_kp=1.5,
        min_creep_mps=0.18,
    )
    assert vx == pytest_approx(0.3)


def test_base_twist_toward_min_creep_still_capped_by_max_linear_mps():
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    vx, vy = base_twist_toward(
        pose,
        (0.05, 0.0),
        max_linear_mps=0.1,
        position_kp=1.5,
        min_creep_mps=0.18,
    )
    assert vx == pytest_approx(0.1)


def test_base_twist_toward_zero_min_creep_is_unchanged_default():
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    vx, vy = base_twist_toward(
        pose, (0.05, 0.0), max_linear_mps=0.5, position_kp=1.5
    )
    assert vx == pytest_approx(0.075)  # pure kp * distance, no floor applied


def test_base_twist_toward_returns_zero_at_target():
    pose = Pose2D(x=2.0, y=-1.0, yaw=1.2)
    vx, vy = base_twist_toward(
        pose, (2.0, -1.0), max_linear_mps=0.5, position_kp=1.5
    )
    assert vx == pytest_approx(0.0, abs=1e-9)
    assert vy == pytest_approx(0.0, abs=1e-9)


def test_pose_reached_true_within_position_tolerance():
    pose = Pose2D(x=0.02, y=0.0, yaw=0.0)
    assert pose_reached(pose, (0.0, 0.0), position_tolerance_m=0.03)


def test_pose_reached_false_outside_position_tolerance():
    pose = Pose2D(x=0.05, y=0.0, yaw=0.0)
    assert not pose_reached(pose, (0.0, 0.0), position_tolerance_m=0.03)


# ---- ProgressWatchdog (handoff sec 18.1b/18.6 W0.4) --------------------- #


def test_progress_watchdog_detects_stall_when_pose_never_changes():
    # A frozen pose (e.g. the base wedged against the door jamb, handoff
    # sec 4.65 hypothesis 2) must be caught within the stall window, not
    # only after the whole tick budget elapses.
    wd = ProgressWatchdog(
        sample_every_ticks=250, stall_window_ticks=1000, min_move_m=0.01
    )
    stalled_at = None
    for tick in range(0, 9000):
        if wd.sample(tick, 1.0, 2.0):
            stalled_at = tick
            break
    assert stalled_at is not None
    # lookback = stall_window_ticks // sample_every_ticks = 4 samples; the
    # 5th sample (tick 1000) is the first one with 4 samples behind it, so
    # that is the earliest possible detection tick.
    assert stalled_at == 1000


def test_progress_watchdog_does_not_trip_when_pose_converges():
    # A base that is genuinely making progress every sample must never be
    # reported as stalled.
    wd = ProgressWatchdog(
        sample_every_ticks=250, stall_window_ticks=1000, min_move_m=0.01
    )
    stalled = False
    x = 0.0
    for tick in range(0, 9000):
        x += 0.0005  # ~0.05 m of travel between samples, well over min_move_m
        if wd.sample(tick, x, 0.0):
            stalled = True
            break
    assert not stalled


def test_progress_watchdog_off_schedule_ticks_are_free():
    wd = ProgressWatchdog(sample_every_ticks=250)
    assert wd.sample(1, 1.0, 1.0) is False
    assert wd.sample(249, 1.0, 1.0) is False
    assert wd.pose_trace == []


def test_progress_watchdog_records_pose_trace_for_diagnosis():
    wd = ProgressWatchdog(
        sample_every_ticks=250, stall_window_ticks=1000, min_move_m=0.01
    )
    for tick in range(0, 1300, 250):
        wd.sample(tick, 1.0, 2.0)
    assert wd.pose_trace == [
        (0, 1.0, 2.0),
        (250, 1.0, 2.0),
        (500, 1.0, 2.0),
        (750, 1.0, 2.0),
        (1000, 1.0, 2.0),
        (1250, 1.0, 2.0),
    ]


def test_pose_reached_checks_yaw_when_target_yaw_given():
    close_yaw = Pose2D(x=0.0, y=0.0, yaw=math.radians(1.0))
    far_yaw = Pose2D(x=0.0, y=0.0, yaw=math.radians(10.0))
    assert pose_reached(
        close_yaw,
        (0.0, 0.0),
        target_yaw=0.0,
        yaw_tolerance_rad=math.radians(3.0),
    )
    assert not pose_reached(
        far_yaw,
        (0.0, 0.0),
        target_yaw=0.0,
        yaw_tolerance_rad=math.radians(3.0),
    )


def test_nearest_obstacle_clearance_m_with_synthetic_obstacles():
    obstacles = [
        {"path": "/A", "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
    ]
    # Outside the box: real Euclidean distance to the nearest edge.
    assert nearest_obstacle_clearance_m(
        (2.0, 0.5), obstacles
    ) == pytest_approx(1.0)
    # Inside the box: 0.0, a hard violation, not "close but clear".
    assert nearest_obstacle_clearance_m(
        (0.5, 0.5), obstacles
    ) == pytest_approx(0.0)


def test_nearest_obstacle_clearance_m_is_inf_when_none_generated():
    assert nearest_obstacle_clearance_m((0.0, 0.0), []) == math.inf


def test_generated_room_obstacles_gate_n5():
    """N5's real gate: the GENERATED artifact must exist, must contain the
    real geometry L3 traced the ROTATE_SPOT stall to (Rectangle012/015,
    SYNC 19) and the kitchen island region (bbox proximity, not a name
    match -- real result: the island prim is "Rectangle127", not anything
    with "island" in its name), the CURRENT ROTATE_SPOT constant must have
    real clearance from it, and the OLD, already-proven-bad (-3.0, -3.1)
    location must NOT -- proving this assertion would have caught the
    original bug, not just that it doesn't flag the current (already-
    fixed) constant."""
    obstacles = load_room_obstacles()
    if not obstacles:
        pytest.skip(
            "assets/derived/room_obstacles.json not generated yet -- run "
            "scripts/task3/generate_room_obstacles.py on a GPU session "
            "before trusting this gate."
        )

    paths = [ob["path"] for ob in obstacles]
    assert any("Rectangle012" in p for p in paths), paths
    assert any("Rectangle015" in p for p in paths), paths

    # The kitchen island prim is NOT named "island" in the USD (real,
    # checked result: it is "Rectangle127", under group "___017", one of
    # several cryptically-named CAD export prims) -- so the real gate is
    # bbox proximity to KITCHEN_ISLAND_BBOX, not a name guess a first
    # draft of this test wrongly assumed.
    from task3_autonomy.navigation import KITCHEN_ISLAND_BBOX

    island_center = (
        (KITCHEN_ISLAND_BBOX["x_min"] + KITCHEN_ISLAND_BBOX["x_max"]) / 2.0,
        (KITCHEN_ISLAND_BBOX["y_min"] + KITCHEN_ISLAND_BBOX["y_max"]) / 2.0,
    )
    assert nearest_obstacle_clearance_m(
        island_center, obstacles
    ) == pytest_approx(0.0, abs=1e-6), (
        "no generated obstacle overlaps KITCHEN_ISLAND_BBOX"
    )

    from task3_autonomy.grasp_transport import ROTATE_SPOT

    current_clearance = nearest_obstacle_clearance_m(ROTATE_SPOT, obstacles)
    assert current_clearance >= BASE_HALF_WIDTH, (
        f"ROTATE_SPOT={ROTATE_SPOT} clearance={current_clearance:.3f}m "
        f"< BASE_HALF_WIDTH={BASE_HALF_WIDTH}m -- the CURRENT constant "
        "would fail this gate"
    )

    old_bad_spot = (-3.0, -3.1)
    old_clearance = nearest_obstacle_clearance_m(old_bad_spot, obstacles)
    assert old_clearance < BASE_HALF_WIDTH, (
        f"old_bad_spot={old_bad_spot} clearance={old_clearance:.3f}m -- "
        "expected this to FAIL the gate (it is the location SYNC 18/19 "
        "already proved stalls against real geometry); if it now passes, "
        "either the room geometry changed or this assertion is too loose "
        "to have caught the original bug"
    )


def test_stance_for_never_grows_past_the_requested_reach_budget():
    """A grasp stance must never be placed farther from the object than the
    radius it asked for.

    `STANCE_REACH_RADIUS_M` is not a preference -- `reach()`'s own grasp
    calibration is measured against exactly this distance, and the constant
    was scaled down to ~0.78 m specifically to keep margin against the
    ~0.855 m measured kinematic ceiling (see its comment). A stance placed
    beyond it is unreachable by construction, which is the failure the
    scaling was introduced to fix.

    Regression: `_rotate_to_clear_island` grew the radius in 0.02 m steps
    whenever no angle cleared the required obstacle margin, silently and up
    to the kinematic limit. Measured, spoon2 at (-4.1, -1.7): the best
    candidate at the requested radius fell 3 mm short of the 0.05 m margin,
    so the stance was pushed to 0.79997 m -- 20 mm past the reach budget --
    with nothing in the return value or the logs saying so.
    """
    from task3_autonomy.navigation import stance_for

    object_xy = (-4.1, -1.7)
    (x, y), _yaw = stance_for(object_xy, "north")
    distance = math.hypot(x - object_xy[0], y - object_xy[1])
    assert distance <= STANCE_REACH_RADIUS_M + 1e-6, (
        f"stance placed {distance:.5f} m from the object, past the "
        f"requested reach budget {STANCE_REACH_RADIUS_M:.5f} m"
    )


def test_stance_for_grows_only_when_a_ceiling_is_asked_for_explicitly():
    """The push path genuinely needs growth and must keep it.

    At `PUSH_STANCE_RADIUS_M` (0.48 m) the ENTIRE annulus lies inside the
    counter's inflated footprint -- 0 of 180 angles clear -- so growing to
    the first legal radius is the only way that stance exists at all. That
    capability stays; it just has to be requested rather than assumed.
    """
    from task3_autonomy.navigation import (
        MEASURED_REACH_LIMIT_M,
        stance_for,
    )

    object_xy = (-4.1, -1.7)
    requested = 0.48

    (nx, ny), _ = stance_for(object_xy, "east", radius_m=requested)
    no_growth = math.hypot(nx - object_xy[0], ny - object_xy[1])
    assert no_growth <= requested + 1e-6

    (gx, gy), _ = stance_for(
        object_xy,
        "east",
        radius_m=requested,
        max_radius_m=MEASURED_REACH_LIMIT_M,
    )
    grown = math.hypot(gx - object_xy[0], gy - object_xy[1])
    assert grown > requested + 1e-6, "explicit ceiling should permit growth"
    assert grown <= MEASURED_REACH_LIMIT_M + 1e-6
