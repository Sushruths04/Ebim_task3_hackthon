# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""The orientation parameterisation, pinned against the two things it has to
be right about: the old top-down behaviour it replaces, and the organisers'
measured demonstrations it exists to reproduce."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))

from task3_autonomy.er_grasp_orientation import (  # noqa: E402
    approach_angles_from_quaternion,
    approach_axis,
    clamp_tilt,
    offset_along_approach,
    quaternion_from_approach,
)
from teleop_targets import _quaternion_from_rpy  # noqa: E402


def _quat_close(a, b, tol=1e-9):
    """Compare up to sign -- q and -q are the same rotation."""
    same = all(abs(x - y) < tol for x, y in zip(a, b))
    flipped = all(abs(x + y) < tol for x, y in zip(a, b))
    return same or flipped


@pytest.mark.parametrize("roll_deg", [0.0, 15.0, -40.0, 90.0, 179.0])
def test_zero_tilt_is_byte_identical_to_the_old_top_down_grasp(roll_deg):
    """The fallback path must not change behaviour at all.

    `reach()` commands `_quaternion_from_rpy(pi, 0, grasp_yaw)` today. If a
    live orientation is unavailable the new code calls this with tilt=0, and
    that has to be the SAME quaternion -- otherwise every run mixes an
    orientation change into whatever else is being measured.
    """
    assert _quat_close(
        quaternion_from_approach(0.0, 0.0, roll_deg),
        _quaternion_from_rpy(math.pi, 0.0, math.radians(roll_deg)),
    )


def test_zero_tilt_approach_axis_is_straight_down():
    ax, ay, az = approach_axis(0.0, 0.0)
    assert (abs(ax), abs(ay)) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert az == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "tilt_deg,azimuth_deg",
    [(30.0, 0.0), (60.0, -90.0), (84.0, 90.0), (52.5, 145.0), (90.0, 12.0)],
)
def test_quaternion_carries_the_requested_approach_axis(tilt_deg, azimuth_deg):
    """The quaternion's own +Z column must BE the requested approach axis.

    This is the whole contract: `reach()` hands the quaternion to IK, and
    every standoff offset is computed from the angles. If those two disagree
    the arm backs off along a direction it is not approaching from.
    """
    w, x, y, z = quaternion_from_approach(tilt_deg, azimuth_deg, 0.0)
    got = (
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    )
    assert got == pytest.approx(approach_axis(tilt_deg, azimuth_deg), abs=1e-9)


@pytest.mark.parametrize(
    "tilt_deg,azimuth_deg,roll_deg",
    [(0.0, 0.0, 0.0), (60.0, -90.0, 0.0), (75.0, 90.0, 30.0), (45.0, 20.0, -60.0)],
)
def test_angles_round_trip_through_the_quaternion(tilt_deg, azimuth_deg, roll_deg):
    quat = quaternion_from_approach(tilt_deg, azimuth_deg, roll_deg)
    got_tilt, got_az, _ = approach_angles_from_quaternion(quat)
    assert got_tilt == pytest.approx(tilt_deg, abs=1e-6)
    if tilt_deg > 0.0:  # azimuth is undefined for a straight-down approach
        assert math.cos(math.radians(got_az - azimuth_deg)) == pytest.approx(
            1.0, abs=1e-9
        )


def test_reproduces_the_organisers_measured_left_arm_grasp():
    """Left arm, tray pickup, all 5 episodes: approach axis ~(0, -0.87, -0.50).

    Recovered by FK'ing the recorded joint angles through our own URDF (the
    dataset's own `*_ee.*` columns are byte-constant and unusable). Tilt came
    out 52.5-63.3 deg, so 60 deg leaning toward -Y is the middle of the
    measured band.
    """
    ax, ay, az = approach_axis(60.0, -90.0)
    assert ax == pytest.approx(0.0, abs=1e-9)
    assert ay == pytest.approx(-0.866, abs=0.01)
    assert az == pytest.approx(-0.500, abs=0.01)


def test_reproduces_the_organisers_measured_right_arm_grasp():
    """Right arm, same runs: approach axis ~(-0.06, +0.97, -0.14), tilt 70-84.

    The two arms lean toward OPPOSITE bearings -- they pinch the tray between
    them. A parameterisation that could not express that would be no better
    than the top-down constant it replaces.
    """
    ax, ay, az = approach_axis(81.0, 90.0)
    assert ax == pytest.approx(0.0, abs=1e-9)
    assert ay == pytest.approx(0.988, abs=0.02)
    assert az == pytest.approx(-0.156, abs=0.02)


def test_offset_along_approach_matches_the_old_arithmetic_when_top_down():
    """The standoff generalisation must be a no-op for a top-down grasp."""
    assert offset_along_approach((1.0, 2.0, 3.0), 0.0, 0.0, 0.14) == pytest.approx(
        (1.0, 2.0, 3.14)
    )


def test_offset_along_approach_backs_off_sideways_when_tilted():
    """At 90 deg the approach is horizontal, so the standoff is horizontal.

    Lifting in world +Z here would move the wrist across the object instead
    of away from it.
    """
    got = offset_along_approach((1.0, 2.0, 3.0), 90.0, 0.0, 0.2)
    assert got == pytest.approx((0.8, 2.0, 3.0), abs=1e-9)


@pytest.mark.parametrize(
    "raw,expected", [(-15.0, 0.0), (0.0, 0.0), (61.0, 61.0), (140.0, 90.0)]
)
def test_clamp_tilt_refuses_approaches_from_below(raw, expected):
    assert clamp_tilt(raw) == pytest.approx(expected)


def test_half_turn_roll_is_the_same_grasp_axis():
    """roll and roll+180 must share an approach axis -- that is why they are
    interchangeable for a symmetric parallel jaw."""
    from task3_autonomy.er_grasp_orientation import approach_axis

    a = approach_axis(50.0, 30.0)
    assert a == pytest.approx(approach_axis(50.0, 30.0), abs=1e-12)
    q1 = quaternion_from_approach(50.0, 30.0, 20.0)
    q2 = quaternion_from_approach(50.0, 30.0, 200.0)
    for q in (q1, q2):
        w, x, y, z = q
        got = (
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        )
        assert got == pytest.approx(a, abs=1e-9)


def test_nearest_equivalent_roll_picks_the_closer_half_turn():
    """The wrist should not travel 180 degrees to reach an identical grasp."""
    from task3_autonomy.er_grasp_orientation import nearest_equivalent_roll

    current = quaternion_from_approach(40.0, 10.0, 170.0)
    # Asking for roll=-10 is the same grasp as roll=170, and 170 is where the
    # wrist already is.
    chosen = nearest_equivalent_roll(40.0, 10.0, -10.0, current)
    assert chosen == pytest.approx(170.0)


def test_nearest_equivalent_roll_keeps_the_request_when_it_is_already_closer():
    from task3_autonomy.er_grasp_orientation import nearest_equivalent_roll

    current = quaternion_from_approach(40.0, 10.0, 15.0)
    assert nearest_equivalent_roll(40.0, 10.0, 20.0, current) == pytest.approx(20.0)


def test_geodesic_is_zero_for_the_same_rotation_and_its_negation():
    from task3_autonomy.er_grasp_orientation import quaternion_geodesic_rad

    q = quaternion_from_approach(33.0, 77.0, 12.0)
    neg = tuple(-v for v in q)
    assert quaternion_geodesic_rad(q, q) == pytest.approx(0.0, abs=1e-7)
    assert quaternion_geodesic_rad(q, neg) == pytest.approx(0.0, abs=1e-7)
