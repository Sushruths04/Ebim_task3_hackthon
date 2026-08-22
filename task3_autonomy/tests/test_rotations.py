# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "common"))

from teleop_targets import _quaternion_from_rpy  # noqa: E402

from task3_autonomy.rotations import rpy_from_quaternion  # noqa: E402

pytest_approx = pytest.approx

# atan2 canonicalizes roll/yaw into (-pi, pi] and asin canonicalizes pitch
# into [-pi/2, pi/2], so the rpy -> quat -> rpy round trip is only exact
# (up to float error) for inputs already inside those open ranges. Keep the
# grid strictly inside -- exact +-pi / +-pi/2 boundaries and the gimbal-lock
# pole are covered by dedicated tests below instead.
_ROLL_YAW_GRID = (
    -3.1,
    -1.5,
    -0.5,
    0.0,
    0.5,
    1.5,
    3.1,
)  # near +-pi, both signs
_PITCH_GRID = (-1.4, -0.5, 0.0, 0.5, 1.4)  # strictly inside +-pi/2


def _assert_rpy_close(actual, expected):
    assert actual[0] == pytest_approx(expected[0], abs=1e-9)
    assert actual[1] == pytest_approx(expected[1], abs=1e-9)
    assert actual[2] == pytest_approx(expected[2], abs=1e-9)


def _assert_quat_close_up_to_sign(actual, expected, *, abs_tol=1e-9):
    same_sign = all(
        a == pytest_approx(e, abs=abs_tol) for a, e in zip(actual, expected)
    )
    flipped_sign = all(
        a == pytest_approx(-e, abs=abs_tol) for a, e in zip(actual, expected)
    )
    assert same_sign or flipped_sign


@pytest.mark.parametrize(
    "roll,pitch,yaw",
    list(itertools.product(_ROLL_YAW_GRID, _PITCH_GRID, _ROLL_YAW_GRID)),
)
def test_rpy_round_trip_through_quaternion(roll, pitch, yaw):
    quat = _quaternion_from_rpy(roll, pitch, yaw)
    recovered = rpy_from_quaternion(quat)
    _assert_rpy_close(recovered, (roll, pitch, yaw))


def test_rpy_round_trip_identity_quaternion():
    quat = _quaternion_from_rpy(0.0, 0.0, 0.0)
    assert quat == pytest_approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)
    assert rpy_from_quaternion(quat) == pytest_approx(
        (0.0, 0.0, 0.0), abs=1e-9
    )
    assert rpy_from_quaternion((1.0, 0.0, 0.0, 0.0)) == pytest_approx(
        (0.0, 0.0, 0.0), abs=1e-9
    )


@pytest.mark.parametrize("pitch_sign", (1.0, -1.0))
def test_rpy_round_trip_near_gimbal_lock_recovers_pitch(pitch_sign):
    # Within 1e-6 rad of the pole: not a domain error, and pitch itself
    # (the only fully-determined component near gimbal lock) round-trips.
    pitch = pitch_sign * (math.pi / 2.0 - 1.0e-6)
    quat = _quaternion_from_rpy(0.3, pitch, -0.4)
    roll, recovered_pitch, yaw = rpy_from_quaternion(quat)
    assert recovered_pitch == pytest_approx(pitch, abs=1e-6)
    assert math.isfinite(roll)
    assert math.isfinite(yaw)
    # This is the exact-pole decomposition applied to a quaternion that is
    # only near the pole (by 1e-6 rad), so the reproduced quaternion is
    # close but not bit-identical -- use a looser tolerance than the exact
    # cases below.
    _assert_quat_close_up_to_sign(
        _quaternion_from_rpy(roll, recovered_pitch, yaw), quat, abs_tol=1e-5
    )


def test_rpy_round_trip_exact_gimbal_lock_does_not_raise():
    quat = _quaternion_from_rpy(0.3, math.pi / 2.0, -0.4)
    roll, pitch, yaw = rpy_from_quaternion(quat)
    assert pitch == pytest_approx(math.pi / 2.0, abs=1e-9)
    assert math.isfinite(roll)
    assert math.isfinite(yaw)
    # Gimbal lock only determines roll - yaw (or roll + yaw); the
    # recovered pair must still reproduce the same rotation.
    _assert_quat_close_up_to_sign(_quaternion_from_rpy(roll, pitch, yaw), quat)


@pytest.mark.parametrize(
    "quat",
    [
        _quaternion_from_rpy(0.2, 0.1, -0.3),
        _quaternion_from_rpy(-1.2, 0.7, 2.5),
        _quaternion_from_rpy(3.0, -0.9, -3.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (2.0, 0.0, 0.0, 0.0),  # non-unit input must still normalize cleanly
    ],
)
def test_quaternion_round_trip_through_rpy(quat):
    roll, pitch, yaw = rpy_from_quaternion(quat)
    recovered_quat = _quaternion_from_rpy(roll, pitch, yaw)
    norm = math.sqrt(sum(component * component for component in quat))
    normalized_quat = tuple(component / norm for component in quat)
    _assert_quat_close_up_to_sign(recovered_quat, normalized_quat)


def test_rpy_from_quaternion_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        rpy_from_quaternion((0.0, 0.0, 0.0, 0.0))
