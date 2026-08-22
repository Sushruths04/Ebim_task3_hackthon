# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Frame conversion for the cuMotion grasp planner.

Only the pure math is tested here -- building a MotionPlanner needs cuRobo,
CUDA and the robot YAML, which belongs in a GPU probe
(`scripts/task3/curobo/probe_motion_planner.py`), not a unit test. What CAN
silently be wrong on CPU is the frame conversion, and that is the exact
failure class that once put ER-2's correctly-identified plate 500 m away.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from task3_autonomy.curobo_grasp_planner import (  # noqa: E402
    quat_conjugate,
    quat_multiply,
    world_pose_to_frame,
)

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_identity_frame_leaves_the_pose_alone():
    p = torch.tensor([1.0, 2.0, 3.0])
    lp, lq = world_pose_to_frame(p, IDENTITY, torch.zeros(3), IDENTITY)
    assert lp.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert lq.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_translated_frame_subtracts_the_origin():
    p = torch.tensor([1.0, 2.0, 3.0])
    lp, _ = world_pose_to_frame(
        p, IDENTITY, torch.tensor([1.0, 1.0, 1.0]), IDENTITY
    )
    assert lp.tolist() == pytest.approx([0.0, 1.0, 2.0])


def test_yawed_frame_rotates_into_the_frame_not_out_of_it():
    """A frame yawed +90 deg sees world +X as its own -Y.

    Getting this inverted is the silent-and-plausible failure: the pose
    lands somewhere real, just wrong.
    """
    yaw90 = torch.tensor(
        [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    )
    lp, _ = world_pose_to_frame(
        torch.tensor([1.0, 0.0, 0.0]), IDENTITY, torch.zeros(3), yaw90
    )
    assert lp.tolist() == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)


def test_round_trip_through_a_general_frame():
    """Converting into a frame and back must return the original pose."""
    frame_pos = torch.tensor([-4.9, -1.3, 0.42])
    frame_quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071])
    p = torch.tensor([-4.2, -1.7, 0.78])
    lp, lq = world_pose_to_frame(p, IDENTITY, frame_pos, frame_quat)
    # back out: world = frame_pos + R(frame_quat) * local
    from task3_autonomy.curobo_grasp_planner import quat_rotate

    back = frame_pos + quat_rotate(frame_quat.unsqueeze(0), lp.unsqueeze(0))[0]
    # 1e-4 = 0.1 mm. float32 plus a 0.7071 literal that is not exactly
    # 1/sqrt(2) leaves ~2.7e-5 here; tighter than that tests the constant,
    # not the conversion.
    assert back.tolist() == pytest.approx(p.tolist(), abs=1e-4)


def test_quat_conjugate_and_multiply_are_inverses():
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])
    out = quat_multiply(quat_conjugate(q.unsqueeze(0)), q.unsqueeze(0))[0]
    assert out.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
