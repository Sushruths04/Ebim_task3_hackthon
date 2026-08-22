# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Grasp ORIENTATION, parameterised the way a perception model can answer it.

Every grasp in this repo has always commanded
``_quaternion_from_rpy(pi, 0, grasp_yaw)`` -- a straight-down approach with a
free wrist spin -- for every object, at every distance. The organisers'
recorded demonstrations say that is not how this robot picks these objects
up.

The evidence (measured 2026-08-14, see ``plans/PROGRESS.md``): both
HuggingFace ``bburdiek/task3_feeding_*`` datasets ship 5 successful episodes
each. Their ``observation.state`` ``*_ee.*`` columns are byte-constant for
every frame and cannot be used, but the joint angles are real, so FK'ing them
through ``assets/derived/mobile_fr3_duo_v0_2.urdf`` recovers the pose that
actually worked. Across all 10 grasps (5 episodes x 2 arms, tray pickup) the
``hand_tcp`` +Z approach axis came out at

    left   ~(0.00, -0.87, -0.50)   tilt 52.5-63.3 deg from straight down
    right  ~(-0.06, +0.97, -0.14)  tilt 70.0-84.3 deg from straight down

i.e. the two arms come in laterally from opposite sides and pinch. A
straight-down command is 52-84 degrees away from that, which is why the reach
residual is FLAT across every spine height instead of shrinking: the arm is
not too short, the wrist pose does not solve.

So orientation needs a real answer per object, per attempt. This module is
the frame conversion for it; ``live_er_grasp.py`` is what asks ER-2.

THE PARAMETERISATION, and why this one:

    tilt_deg     angle of the approach axis away from straight-down (0 =
                 top-down, 90 = horizontal)
    azimuth_deg  which way it leans, as a world-frame compass bearing:
                 0 = +X, 90 = +Y
    roll_deg     spin of the jaws about the approach axis -- exactly the old
                 ``grasp_yaw`` and it keeps that meaning

Three scalars, each independently meaningful, each answerable in words by a
vision model looking at a picture ("come in from the side, from the left, and
hold it across"). A raw quaternion is not answerable that way and a model
asked for one returns noise.

**At ``tilt_deg == 0`` this reproduces ``_quaternion_from_rpy(pi, 0, roll)``
exactly**, so a caller that cannot get an orientation keeps the old behaviour
bit for bit and nothing downstream has to special-case the fallback.
"""

from __future__ import annotations

import math

QuaternionWxyz = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]

# The straight-down approach the whole repo used to assume, kept as the
# explicit identity of this parameterisation rather than as an implicit 0.
TOP_DOWN_TILT_DEG = 0.0

# Bounds accepted from a perception model before its answer is refused. A
# tilt past 90 deg would have the gripper approaching from BELOW the object,
# which cannot be right for anything resting on a counter, and a negative
# tilt is the same pose as its positive twin with azimuth flipped 180 -- so
# the ambiguous half of the range is rejected rather than silently folded.
MIN_TILT_DEG = 0.0
MAX_TILT_DEG = 90.0


def _rotation_to_quaternion(r: list[list[float]]) -> QuaternionWxyz:
    """Rotation matrix -> (w, x, y, z), Shepperd's method.

    Branching on the largest diagonal term rather than always using the
    trace: the trace form divides by ``sqrt(1 + trace)``, which goes to zero
    for a 180-degree rotation, and a 180-degree rotation is not a corner case
    here -- ``pi`` roll IS the top-down grasp this module has to reproduce
    exactly.
    """
    m00, m01, m02 = r[0]
    m10, m11, m12 = r[1]
    m20, m21, m22 = r[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def approach_axis(tilt_deg: float, azimuth_deg: float) -> Vec3:
    """Unit vector the gripper travels ALONG to reach the object (tcp +Z).

    Straight down is ``(0, 0, -1)``; tilting leans the far end of that vector
    toward the ``azimuth_deg`` bearing.
    """
    tilt = math.radians(tilt_deg)
    az = math.radians(azimuth_deg)
    return (
        math.sin(tilt) * math.cos(az),
        math.sin(tilt) * math.sin(az),
        -math.cos(tilt),
    )


def quaternion_from_approach(
    tilt_deg: float,
    azimuth_deg: float,
    roll_deg: float = 0.0,
) -> QuaternionWxyz:
    """``(tilt, azimuth, roll)`` -> a wrist quaternion in world frame, wxyz.

    Built as ``Rot(n, tilt) @ Rz(roll) @ Rx(pi)``. The right-hand factor is
    the old top-down grasp; the left-hand factor is the rotation that carries
    its ``(0, 0, -1)`` approach axis onto ``approach_axis(tilt, azimuth)``,
    taken about ``n = (sin az, -cos az, 0)`` -- the horizontal axis
    perpendicular to the lean, which is the shortest such rotation and so
    does not spin the jaws as a side effect.

    With ``tilt_deg == 0`` the left factor is the identity and the result is
    ``_quaternion_from_rpy(pi, 0, roll)`` to floating-point equality.
    """
    roll = math.radians(roll_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    # Rz(roll) @ Rx(pi): columns are the old top-down frame's axes.
    base = [
        [cr, sr, 0.0],
        [sr, -cr, 0.0],
        [0.0, 0.0, -1.0],
    ]

    tilt = math.radians(tilt_deg)
    az = math.radians(azimuth_deg)
    nx, ny, nz = math.sin(az), -math.cos(az), 0.0
    c, s = math.cos(tilt), math.sin(tilt)
    one_c = 1.0 - c
    # Rodrigues for the tilt about n.
    lean = [
        [c + nx * nx * one_c, nx * ny * one_c - nz * s, nx * nz * one_c + ny * s],
        [ny * nx * one_c + nz * s, c + ny * ny * one_c, ny * nz * one_c - nx * s],
        [nz * nx * one_c - ny * s, nz * ny * one_c + nx * s, c + nz * nz * one_c],
    ]
    return _rotation_to_quaternion(_matmul(lean, base))


def approach_angles_from_quaternion(
    quaternion: QuaternionWxyz,
) -> tuple[float, float, float]:
    """Inverse of :func:`quaternion_from_approach`, in degrees.

    Exists so a run can LOG what orientation it actually commanded in the
    same three numbers the demonstrations were measured in -- comparing a
    commanded quaternion against "52-84 degrees of tilt" is otherwise a
    manual conversion every time, and that is how a wrong orientation went
    unnoticed for as long as it did.
    """
    w, x, y, z = quaternion
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    # Third column of the rotation matrix: the approach axis.
    ax = 2.0 * (x * z + y * w)
    ay = 2.0 * (y * z - x * w)
    az_ = 1.0 - 2.0 * (x * x + y * y)
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -az_))))
    azimuth_deg = math.degrees(math.atan2(ay, ax)) if (ax or ay) else 0.0
    # Undo the lean, then read the residual spin off the first column.
    undo = quaternion_from_approach(-tilt_deg, azimuth_deg, 0.0)
    uw, ux, uy, uz = undo
    # roll is whatever spin remains once the tilt is removed; recovering it
    # through the matrix is cheaper than composing quaternions here.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + z * w)
    roll_deg = math.degrees(math.atan2(r10, r00))
    del uw, ux, uy, uz
    return tilt_deg, azimuth_deg, roll_deg


def quaternion_geodesic_rad(
    a: QuaternionWxyz, b: QuaternionWxyz
) -> float:
    """Shortest rotation angle between two orientations, in radians.

    Uses ``abs(dot)`` because ``q`` and ``-q`` are the same rotation, so the
    unsigned form is the only one that measures what a joint would actually
    have to travel.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("orientation quaternions must be nonzero")
    return 2.0 * math.acos(min(1.0, abs(dot) / (na * nb)))


def nearest_equivalent_roll(
    tilt_deg: float,
    azimuth_deg: float,
    roll_deg: float,
    current_quaternion_wxyz: QuaternionWxyz,
) -> float:
    """Pick whichever of ``roll`` / ``roll + 180`` the wrist can reach sooner.

    A parallel jaw is SYMMETRIC under a half turn about its approach axis:
    rolling 180 degrees swaps which finger is which and leaves the grasp
    physically identical. So the two rolls are the same grasp and we are free
    to command whichever one is closer to where the wrist already is.

    That freedom is worth taking because of what limits this arm. Joint 7 has
    a 12 N*m effort limit against a damping of 500, capping it at
    ``12/500 = 0.024 rad/s``; `arms.reach` gets 4 s, so it can only travel
    ~0.1 rad before time runs out. Measured on run 11: joint 7 was commanded
    0.42 rad away, moved almost not at all, and left every reach 2-3 cm short
    -- the plateau in `position_error_trace` that no amount of extra servo
    time closed. Halving the wrist travel by choosing the equivalent roll is
    free accuracy, and unlike a gains change it cannot destabilise anything.

    Returns the chosen roll in degrees.
    """
    candidates = (roll_deg, roll_deg + 180.0)
    best_roll = roll_deg
    best_distance = None
    for candidate in candidates:
        quaternion = quaternion_from_approach(
            tilt_deg, azimuth_deg, candidate
        )
        distance = quaternion_geodesic_rad(
            quaternion, current_quaternion_wxyz
        )
        if best_distance is None or distance < best_distance:
            best_distance, best_roll = distance, candidate
    # Keep the reported angle in a readable range; the quaternion is
    # unaffected by the wrap.
    return (best_roll + 180.0) % 360.0 - 180.0


def clamp_tilt(tilt_deg: float) -> float:
    """Hold a model-supplied tilt inside the physically meaningful range."""
    return max(MIN_TILT_DEG, min(MAX_TILT_DEG, float(tilt_deg)))


def offset_along_approach(
    point: Vec3,
    tilt_deg: float,
    azimuth_deg: float,
    distance_m: float,
) -> Vec3:
    """Back the gripper off ``distance_m`` from ``point``, along its approach.

    Every standoff/pregrasp in ``reach()`` was written as a change in world
    ``+Z``, which is only "backing away from the object" while the approach
    is straight down. At a 70-degree tilt, lifting in world Z moves the wrist
    mostly SIDEWAYS relative to its own approach and drags the jaws across
    the object -- the same class of error as the pregrasp lateral slide
    already documented in ``reach()``.

    At ``tilt_deg == 0`` the approach axis is ``(0, 0, -1)`` and this returns
    ``point + (0, 0, distance_m)``, i.e. exactly the old arithmetic.
    """
    ax, ay, az_ = approach_axis(tilt_deg, azimuth_deg)
    return (
        point[0] - ax * distance_m,
        point[1] - ay * distance_m,
        point[2] - az_ * distance_m,
    )
