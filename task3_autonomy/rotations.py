# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Invert ``_quaternion_from_rpy`` from scripts/common/teleop_targets.py.

The reach() skill (task3_autonomy/skills.py) reads the measured end-effector
pose out of Isaac Sim as a scalar-first quaternion, but the teleop boundary
(scripts/common/teleop_targets.py) speaks roll/pitch/yaw deltas. Converting a
measured quaternion into an RPY target needs the exact inverse of that
module's private ``_quaternion_from_rpy`` helper, matching its rotation order
(intrinsic roll-then-pitch-then-yaw, i.e. q = q_yaw * q_pitch * q_roll) and
its scalar-first (w, x, y, z) quaternion convention. No torch/numpy/Isaac
imports here -- unit-testable on CPU, same rule as task3_autonomy/navigation.py.
"""

from __future__ import annotations

import math

QuaternionWxyz = tuple[float, float, float, float]

# Above this, treat asin's argument as saturated (both floating-point
# overshoot past +-1 and genuine near-gimbal-lock poses) rather than let
# asin() raise a domain error.
_GIMBAL_LOCK_THRESHOLD = 1.0 - 1.0e-9


def rpy_from_quaternion(quat: QuaternionWxyz) -> tuple[float, float, float]:
    """Recover (roll, pitch, yaw) radians from a scalar-first quaternion.

    Inverts ``_quaternion_from_rpy(roll, pitch, yaw)`` from
    scripts/common/teleop_targets.py, so ``rpy_from_quaternion`` composed
    with that helper round-trips (up to the sign ambiguity of quaternions
    and the +-pi / +-pi/2 wrap of Euler angles).

    Near pitch = +-90 degrees the roll/yaw decomposition is singular
    (gimbal lock: only roll+yaw or roll-yaw is determined). This clamps the
    pitch asin() argument instead of raising, and returns the roll/yaw pair
    that ``atan2`` produces at the (clamped) pole -- a valid, if non-unique,
    decomposition.
    """
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must have non-zero finite norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))

    if abs(sinp) >= _GIMBAL_LOCK_THRESHOLD:
        # At the pole only one combination of roll and yaw is observable
        # (roll - yaw at the north pole, roll + yaw at the south pole; both
        # reduce to the same atan2(x, w) expression). The standard/atan2
        # formulas below degenerate to atan2(0, 0) here and silently lose
        # that information, so special-case it: fold everything into roll
        # and fix yaw at zero -- one of the infinitely many equivalent
        # decompositions, but one that reconstructs the original rotation.
        pitch = math.copysign(math.pi / 2.0, sinp)
        combined = 2.0 * math.atan2(x, w)
        roll = math.atan2(math.sin(combined), math.cos(combined))
        yaw = 0.0
        return (roll, pitch, yaw)

    pitch = math.asin(sinp)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)
