# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the M1 real-backend pure math (task3_pipeline/
sim_camera_perception.py) -- pinhole back-projection and the camera-to-
world transform. This is the part GATE M1's measured error distribution
actually depends on, so it is checked here independent of any model or
Isaac Sim session (which is not verified as of the commit that added
this file -- see the module docstring).

Run: python -m pytest task3_pipeline/tests/test_sim_camera_perception.py -q
"""

from __future__ import annotations

import math

import pytest

from task3_pipeline.sim_camera_perception import (
    CameraIntrinsics,
    back_project,
    camera_point_to_world,
    detection_to_world_position,
    intrinsics_from_fov,
    perceived_pose_from_detection,
)
from task3_pipeline.sim_camera_perception import (
    _look_at_quaternion_wxyz as look_at_quaternion_wxyz,
)

INTRINSICS = CameraIntrinsics(
    fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
)


def test_back_project_center_pixel_lies_on_the_optical_axis():
    # The principal point at any depth must back-project to (0, 0, depth)
    # in the camera frame -- straight ahead, no lateral offset.
    x, y, z = back_project(INTRINSICS.cx, INTRINSICS.cy, 2.0, INTRINSICS)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(2.0)


def test_back_project_off_center_pixel_matches_the_pinhole_formula():
    # A pixel 50 px right and 25 px down of center, at depth 4 m:
    # x = (u - cx) * depth / fx = 50 * 4 / 500 = 0.4
    # y = (v - cy) * depth / fy = 25 * 4 / 500 = 0.2
    x, y, z = back_project(370.0, 265.0, 4.0, INTRINSICS)
    assert x == pytest.approx(0.4)
    assert y == pytest.approx(0.2)
    assert z == pytest.approx(4.0)


def test_back_project_rejects_nonpositive_depth():
    with pytest.raises(ValueError):
        back_project(320.0, 240.0, 0.0, INTRINSICS)
    with pytest.raises(ValueError):
        back_project(320.0, 240.0, -1.0, INTRINSICS)


def test_camera_point_to_world_identity_orientation_is_pure_translation():
    point_camera = (1.0, 2.0, 3.0)
    camera_pos = (10.0, 20.0, 30.0)
    identity = (1.0, 0.0, 0.0, 0.0)
    world = camera_point_to_world(point_camera, camera_pos, identity)
    assert world == pytest.approx((11.0, 22.0, 33.0))


def test_camera_point_to_world_90deg_about_z_matches_hand_derived_result():
    # q = (cos45, 0, 0, sin45) rotates (1, 0, 0) -> (0, 1, 0) by the
    # standard right-hand quaternion rotation formula -- verified by
    # hand (Rodrigues' expansion) in the commit that added this test,
    # not just re-deriving the same code under test.
    half = math.radians(45.0)
    q = (math.cos(half), 0.0, 0.0, math.sin(half))
    world = camera_point_to_world((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), q)
    assert world[0] == pytest.approx(0.0, abs=1e-9)
    assert world[1] == pytest.approx(1.0, abs=1e-9)
    assert world[2] == pytest.approx(0.0, abs=1e-9)


def test_camera_point_to_world_normalizes_a_nonunit_quaternion():
    # A quaternion with the right direction but wrong magnitude must
    # still rotate correctly -- silently trusting an unnormalized
    # quaternion is exactly the kind of arithmetic bug this project
    # keeps finding (ACTIVE_BRIEF sec 3).
    half = math.radians(45.0)
    q_unit = (math.cos(half), 0.0, 0.0, math.sin(half))
    q_scaled = tuple(3.0 * c for c in q_unit)
    world = camera_point_to_world((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), q_scaled)
    assert world[1] == pytest.approx(1.0, abs=1e-9)


def test_camera_point_to_world_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        camera_point_to_world(
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
        )


def test_detection_to_world_position_composes_backproject_and_transform():
    identity = (1.0, 0.0, 0.0, 0.0)
    camera_pos = (5.0, 5.0, 1.0)
    world = detection_to_world_position(
        INTRINSICS.cx, INTRINSICS.cy, 2.0, INTRINSICS, camera_pos, identity
    )
    # Center pixel at depth 2 -> camera-frame (0,0,2) -> + camera_pos.
    assert world == pytest.approx((5.0, 5.0, 3.0))


def test_perceived_pose_from_detection_matches_ground_truth_perception_shape():
    identity = (1.0, 0.0, 0.0, 0.0)
    pose = perceived_pose_from_detection(
        INTRINSICS.cx,
        INTRINSICS.cy,
        1.5,
        INTRINSICS,
        (0.0, 0.0, 0.0),
        identity,
        confidence=0.87,
    )
    assert pose.position == pytest.approx((0.0, 0.0, 1.5))
    assert pose.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert pose.confidence == pytest.approx(0.87)
    assert pose.visible is True


def test_camera_intrinsics_rejects_nonpositive_focal_length():
    with pytest.raises(ValueError):
        CameraIntrinsics(
            fx=0.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480
        )


def test_camera_intrinsics_rejects_nonpositive_image_size():
    with pytest.raises(ValueError):
        CameraIntrinsics(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=0, height=480
        )


def test_intrinsics_from_fov_center_matches_image_midpoint():
    intr = intrinsics_from_fov((640, 480), 60.0)
    assert intr.cx == pytest.approx(320.0)
    assert intr.cy == pytest.approx(240.0)
    assert intr.width == 640
    assert intr.height == 480


def test_intrinsics_from_fov_matches_hand_computed_focal_length():
    # fx = (width/2) / tan(fov/2). For a 90 deg horizontal FOV, tan(45) = 1,
    # so fx must equal exactly width/2 -- a clean, hand-checkable case.
    intr = intrinsics_from_fov((640, 480), 90.0)
    assert intr.fx == pytest.approx(320.0)
    assert intr.fy == pytest.approx(320.0)


def test_intrinsics_from_fov_wider_fov_gives_shorter_focal_length():
    narrow = intrinsics_from_fov((640, 480), 30.0)
    wide = intrinsics_from_fov((640, 480), 90.0)
    assert wide.fx < narrow.fx


def test_look_at_quaternion_straight_down_is_identity():
    # Camera above the origin looking straight down (-z world) is exactly
    # the camera's own default forward (-z in its local frame) -- no
    # rotation needed.
    q = look_at_quaternion_wxyz((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    assert q[0] == pytest.approx(1.0, abs=1e-9)
    assert q[1] == pytest.approx(0.0, abs=1e-9)
    assert q[2] == pytest.approx(0.0, abs=1e-9)
    assert q[3] == pytest.approx(0.0, abs=1e-9)


def test_look_at_quaternion_rotates_forward_vector_correctly():
    # Whatever quaternion comes out, applying it to the camera's local
    # forward (0, 0, -1) must reproduce the actual normalized look
    # direction -- this is the real end-to-end property that matters,
    # not the quaternion's specific component values.
    position = (2.0, 0.0, 0.0)
    look_at = (2.0, 5.0, 0.0)  # forward should be world +y
    q = look_at_quaternion_wxyz(position, look_at)
    rotated = camera_point_to_world((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), q)
    assert rotated[0] == pytest.approx(0.0, abs=1e-9)
    assert rotated[1] == pytest.approx(1.0, abs=1e-9)
    assert rotated[2] == pytest.approx(0.0, abs=1e-9)


def test_look_at_quaternion_rejects_coincident_position_and_target():
    with pytest.raises(ValueError):
        look_at_quaternion_wxyz((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
