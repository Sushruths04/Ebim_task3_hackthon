# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for perception_grasp.py's pure math (REV16 Phase C.3: "CPU-
unit-test project_to_world against known synthetic matrices before
spending a single GPU episode on it -- a projection sign error is the
classic way to burn six hours"). The GPU-only piece
(``screen_grasp_candidate_ik``, which calls into ``world.arms``/USD) is
exercised on GPU only, same scoping rule
``task3_autonomy/tests/test_perception_targets.py`` already states for
the sibling module.

Run: python -m pytest task3_autonomy/tests/test_perception_grasp.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task3_autonomy.perception_grasp import (  # noqa: E402
    GraspCandidate,
    grasp_point_from_mask,
    mask_centroid_px,
    mask_principal_axes_px,
    meters_per_pixel,
    project_to_pixel,
    project_to_world,
    segment,
)
from task3_autonomy.perception_targets import IkScreenResult  # noqa: E402

IDENTITY_4X4 = np.eye(4).tolist()

# proj[0][0] = proj[1][1] = 1.0, proj[2][3] = -1.0 -- the minimal
# perspective matrix making w = -cam_z = depth_m in the row-vector *
# matrix convention this module documents (derivation in the module
# docstring; this is the concrete matrix that makes the algebra real).
SIMPLE_PROJ = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.0],
    [0.0, 0.0, 0.0, 0.0],
]


def _translation_view(tx: float, ty: float, tz: float) -> list[list[float]]:
    """Unrotated camera at world position (tx, ty, tz): cam_pt = world_pt
    - camera_position, expressed as a row3-translation matrix (world_pt
    @ M convention)."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-tx, -ty, -tz, 1.0],
    ]


def _rotate_z_90_view() -> list[list[float]]:
    """Pure 90-degree rotation about world Z, no translation -- an
    orthonormal (and therefore easily hand-checkable) matrix."""
    return [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------
# project_to_world / project_to_pixel -- hand-derived cases
# ---------------------------------------------------------------------


def test_project_to_world_center_pixel_lies_on_the_optical_axis():
    x, y, z = project_to_world(
        50.0, 50.0, 10.0, IDENTITY_4X4, SIMPLE_PROJ, 100, 100
    )
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(-10.0)


def test_project_to_world_off_axis_matches_hand_derived_result():
    # u=100 (right edge) -> ndc_x=1.0; v=50 (vertical center) -> ndc_y=0.
    # cam_x = ndc_x * depth / proj[0][0] = 1.0 * 10 / 1.0 = 10.
    x, y, z = project_to_world(
        100.0, 50.0, 10.0, IDENTITY_4X4, SIMPLE_PROJ, 100, 100
    )
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(-10.0)


def test_project_to_world_scales_with_proj_focal_term():
    # Same pixel/depth as above but proj[0][0]=2.0 halves cam_x --
    # isolates the proj[0][0] division term from the NDC term.
    proj = [row[:] for row in SIMPLE_PROJ]
    proj[0][0] = 2.0
    x, _y, _z = project_to_world(
        100.0, 50.0, 10.0, IDENTITY_4X4, proj, 100, 100
    )
    assert x == pytest.approx(5.0)


def test_project_to_world_rejects_nonpositive_depth():
    with pytest.raises(ValueError):
        project_to_world(50.0, 50.0, 0.0, IDENTITY_4X4, SIMPLE_PROJ, 100, 100)
    with pytest.raises(ValueError):
        project_to_world(50.0, 50.0, -1.0, IDENTITY_4X4, SIMPLE_PROJ, 100, 100)


def test_project_to_world_accepts_flat_16_element_matrices():
    # camera_params annotators commonly hand back flat arrays, not
    # nested 4x4 ones -- both must work identically.
    flat_view = [v for row in IDENTITY_4X4 for v in row]
    flat_proj = [v for row in SIMPLE_PROJ for v in row]
    nested = project_to_world(
        100.0, 50.0, 10.0, IDENTITY_4X4, SIMPLE_PROJ, 100, 100
    )
    flat = project_to_world(100.0, 50.0, 10.0, flat_view, flat_proj, 100, 100)
    assert nested == pytest.approx(flat)


# ---------------------------------------------------------------------
# round trip: project_to_pixel(project_to_world(...)) recovers the input
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "u,v,depth",
    [(50.0, 50.0, 10.0), (10.0, 90.0, 3.0), (95.0, 5.0, 25.0)],
)
def test_round_trip_translated_camera(u, v, depth):
    view = _translation_view(2.0, -1.5, 0.5)
    world = project_to_world(u, v, depth, view, SIMPLE_PROJ, 100, 100)
    u2, v2, depth2 = project_to_pixel(world, view, SIMPLE_PROJ, 100, 100)
    assert u2 == pytest.approx(u, abs=1e-6)
    assert v2 == pytest.approx(v, abs=1e-6)
    assert depth2 == pytest.approx(depth, abs=1e-6)


@pytest.mark.parametrize(
    "u,v,depth",
    [(50.0, 50.0, 10.0), (10.0, 90.0, 3.0), (95.0, 5.0, 25.0)],
)
def test_round_trip_rotated_camera(u, v, depth):
    view = _rotate_z_90_view()
    world = project_to_world(u, v, depth, view, SIMPLE_PROJ, 100, 100)
    u2, v2, depth2 = project_to_pixel(world, view, SIMPLE_PROJ, 100, 100)
    assert u2 == pytest.approx(u, abs=1e-6)
    assert v2 == pytest.approx(v, abs=1e-6)
    assert depth2 == pytest.approx(depth, abs=1e-6)


# ---------------------------------------------------------------------
# meters_per_pixel
# ---------------------------------------------------------------------


def test_meters_per_pixel_matches_hand_derived_formula():
    # full_width_at_depth = 2 * depth / proj[0][0] = 2*10/1.0 = 20;
    # mpp = 20 / 100 = 0.2.
    mpp = meters_per_pixel(SIMPLE_PROJ, 10.0, 100)
    assert mpp == pytest.approx(0.2)


def test_meters_per_pixel_scales_inversely_with_proj_focal_term():
    proj = [row[:] for row in SIMPLE_PROJ]
    proj[0][0] = 2.0
    mpp = meters_per_pixel(proj, 10.0, 100)
    assert mpp == pytest.approx(0.1)


def test_meters_per_pixel_rejects_nonpositive_depth():
    with pytest.raises(ValueError):
        meters_per_pixel(SIMPLE_PROJ, 0.0, 100)


# ---------------------------------------------------------------------
# mask helpers
# ---------------------------------------------------------------------


def test_mask_centroid_px_matches_hand_computed_center():
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 4:16] = True  # rows 8-11, cols 4-15 -> centroid (9.5, 5+5.5)
    u, v = mask_centroid_px(mask)
    assert u == pytest.approx(9.5)
    assert v == pytest.approx(9.5)


def test_mask_centroid_px_raises_on_empty_mask():
    with pytest.raises(ValueError):
        mask_centroid_px(np.zeros((10, 10), dtype=bool))


def test_mask_principal_axes_px_horizontal_rectangle_is_x_major():
    mask = np.zeros((20, 60), dtype=bool)
    mask[9:12, 5:56] = True  # 3 rows tall, 51 cols wide -> x is major
    major, minor, major_extent, minor_extent = mask_principal_axes_px(mask)
    assert abs(major[0]) > abs(major[1])  # major axis ~ horizontal
    assert abs(minor[1]) > abs(minor[0])  # minor axis ~ vertical
    assert major_extent > minor_extent
    assert major_extent == pytest.approx(50.0, abs=1.0)
    assert minor_extent == pytest.approx(2.0, abs=1.0)


def test_mask_principal_axes_px_vertical_rectangle_is_y_major():
    mask = np.zeros((60, 20), dtype=bool)
    mask[5:56, 9:12] = True  # tall and narrow -> y is major
    major, _minor, major_extent, minor_extent = mask_principal_axes_px(mask)
    assert abs(major[1]) > abs(major[0])
    assert major_extent > minor_extent


def test_mask_principal_axes_px_raises_when_too_small():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError):
        mask_principal_axes_px(mask)


# ---------------------------------------------------------------------
# segment()
# ---------------------------------------------------------------------


def test_segment_matches_ids_by_prim_substring():
    data = np.array([[1, 1, 2], [1, 2, 2], [0, 0, 2]])
    seg_data = {
        "data": data,
        "info": {"idToLabels": {"1": "/World/cup_01", "2": "/World/table"}},
    }
    mask = segment(seg_data, "cup")
    expected = data == 1
    assert np.array_equal(mask, expected)


def test_segment_returns_all_false_when_nothing_matches():
    data = np.array([[1, 1], [2, 2]])
    seg_data = {
        "data": data,
        "info": {"idToLabels": {"1": "/World/table", "2": "/World/floor"}},
    }
    mask = segment(seg_data, "cup")
    assert mask.dtype == bool
    assert mask.shape == data.shape
    assert not mask.any()


# ---------------------------------------------------------------------
# grasp_point_from_mask -- integration of the pieces above
# ---------------------------------------------------------------------


def _synthetic_mask_and_depth(
    shape=(100, 100), rows=slice(45, 55), cols=slice(20, 80), depth_val=5.0
):
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    depth = np.full(shape, np.nan)
    depth[rows, cols] = depth_val
    return mask, depth


def test_grasp_point_from_mask_returns_none_for_empty_mask():
    mask = np.zeros((50, 50), dtype=bool)
    depth = np.ones((50, 50))
    result = grasp_point_from_mask(
        "cup", mask, depth, IDENTITY_4X4, SIMPLE_PROJ, 50, 50
    )
    assert result is None


def test_grasp_point_from_mask_returns_none_when_depth_all_invalid():
    mask, _depth = _synthetic_mask_and_depth()
    depth = np.full(mask.shape, np.nan)
    result = grasp_point_from_mask(
        "cup", mask, depth, IDENTITY_4X4, SIMPLE_PROJ, 100, 100
    )
    assert result is None


def test_grasp_point_from_mask_adds_clearance_in_world_z():
    mask, depth = _synthetic_mask_and_depth(depth_val=5.0)
    result = grasp_point_from_mask(
        "cup",
        mask,
        depth,
        IDENTITY_4X4,
        SIMPLE_PROJ,
        100,
        100,
        clearance_m=0.02,
    )
    assert result is not None
    # Identity view, nearest depth = 5.0 -> surface world z = -5.0
    # (project_to_world's own center/off-axis tests establish this sign
    # convention); grasp z must be surface z + clearance, exactly.
    assert result.xyz[2] == pytest.approx(-5.0 + 0.02, abs=1e-3)


def test_grasp_point_from_mask_yaw_is_perpendicular_to_major_axis():
    # Mask is wide in x (major axis ~horizontal, angle 0) -> yaw should
    # be pi/2 modulo pi (PCA eigenvector sign is arbitrary).
    mask, depth = _synthetic_mask_and_depth()
    result = grasp_point_from_mask(
        "cup", mask, depth, IDENTITY_4X4, SIMPLE_PROJ, 100, 100
    )
    assert result is not None
    wrapped = result.yaw_rad % math.pi
    assert wrapped == pytest.approx(math.pi / 2, abs=0.05)


def test_grasp_point_from_mask_width_ok_toggles_with_gripper_opening():
    mask, depth = _synthetic_mask_and_depth(
        rows=slice(45, 55), cols=slice(20, 80)
    )
    narrow_limit = grasp_point_from_mask(
        "cup",
        mask,
        depth,
        IDENTITY_4X4,
        SIMPLE_PROJ,
        100,
        100,
        gripper_max_opening_m=1e-6,
    )
    wide_limit = grasp_point_from_mask(
        "cup",
        mask,
        depth,
        IDENTITY_4X4,
        SIMPLE_PROJ,
        100,
        100,
        gripper_max_opening_m=1e6,
    )
    assert narrow_limit is not None and wide_limit is not None
    assert narrow_limit.width_m == pytest.approx(wide_limit.width_m)
    assert narrow_limit.width_ok is False
    assert wide_limit.width_ok is True


def test_grasp_point_from_mask_pixel_count_matches_mask_sum():
    mask, depth = _synthetic_mask_and_depth()
    result = grasp_point_from_mask(
        "cup", mask, depth, IDENTITY_4X4, SIMPLE_PROJ, 100, 100
    )
    assert result is not None
    assert result.mask_pixel_count == int(mask.sum())


# ---------------------------------------------------------------------
# GraspCandidate -- mirrors ScreenedCandidate's own tests
# ---------------------------------------------------------------------


def test_grasp_candidate_any_side_feasible_and_best_norm():
    cand = GraspCandidate(
        object="cup",
        xyz=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        width_m=0.05,
        width_ok=True,
        mask_pixel_count=100,
    )
    cand.sides = [
        IkScreenResult(
            side="left", ik_feasible=False, target_norm_from_base_m=0.2
        ),
        IkScreenResult(
            side="right", ik_feasible=True, target_norm_from_base_m=0.6
        ),
    ]
    assert cand.any_side_feasible is True
    assert cand.best_feasible_norm_m == 0.6


def test_grasp_candidate_no_feasible_sides():
    cand = GraspCandidate(
        object="cup",
        xyz=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
        width_m=0.05,
        width_ok=True,
        mask_pixel_count=100,
    )
    assert cand.any_side_feasible is False
    assert cand.best_feasible_norm_m is None
