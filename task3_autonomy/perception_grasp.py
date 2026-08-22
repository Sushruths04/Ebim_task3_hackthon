# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REV16 Phase C -- perception-driven grasp POINT selection: segment the
target object, pick a grasp point ON the mask via its principal axis,
back-project pixel + depth to a 3-D world point using the camera's own
view/projection matrices, and IK-screen the candidate.

**Not the same problem `task3_pipeline/sim_camera_perception.py` solves.**
That module (M1, already built) answers WHERE AN OBJECT IS -- an OWL-ViT
bounding-box centroid, position only, feeding M3's RL observation -- via
pinhole back-projection from a plain focal-length/principal-point
intrinsics model. This module answers WHERE ON THE OBJECT to grasp: the
mask's principal axis picks a point off-centre and perpendicular to the
object's long axis (a rim/handle, not a centre pinch), plus a minor-axis
WIDTH CHECK the position-only module has no reason to compute. REV13 T6's
per-tick telemetry (EE climbs 14cm over a `lift()` call, object z moves
0.02mm the entire time, every prior "held" signal reads green) is
consistent with exactly the failure this check exists to catch: a centre
pinch closing on an object wider than the gripper can actually span. That
is a hypothesis this module lets a future session test, not a proven
cause -- do not claim more than the measured back-projection error until
a real GPU episode confirms it (REV16 Phase C.3).

Default-off, new module: nothing in the live grasp path imports this yet
(same rule `task3_autonomy/perception_targets.py`'s own docstring
states). Real kinematic IK screening is NOT reimplemented here --
`screen_candidate_ik` (`task3_autonomy/perception_targets.py`) is reused
directly, the same solver `DualArmController.command()` calls every
production tick.

Camera math: the view/projection matrices below are the SAME ones
`omni.replicator.core`'s `camera_params` annotator exposes
(`cameraViewTransform`/`cameraProjection`, row-vector * matrix
convention, camera looks down -Z in its own space) -- the raw-array form
of the identical algebra `scripts/task3/probe_gemini_er_vs_ground_truth.
py::SceneCamera.project`/`unproject_at_depth` already uses via a USD
`Gf.Camera` frustum object. `project_to_world`/`project_to_pixel` below
generalize that from a frustum object to matrix arrays because
`camera_params` gives arrays, not a frustum -- the round-trip test in
`task3_autonomy/tests/test_perception_grasp.py` is the actual proof of
that generalization, not the derivation comment.

Segmentation source today is Isaac Sim's ground-truth
`instance_segmentation_fast` annotator (REV16 C.0: "you do not need to
train a segmentation model"). `segment()`'s signature takes that
annotator's raw dict and must not be assumed by any caller downstream --
swapping in a learned segmenter later changes only this one function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from task3_autonomy.perception_targets import (
    IkScreenResult,
    screen_candidate_ik,
)

# Franka Hand (FR3) documented physical max finger-to-finger opening.
# This is an EXTERNAL hardware spec, not a number measured or derived
# anywhere in this repo -- `arms.py`'s `GRIPPER_OPEN_RAD = 0.9` is a
# joint angle, and no metres-per-radian conversion for it is recorded
# anywhere in this codebase. Treat this constant as unverified until a
# real GPU episode measures the actual open-gripper span and confirms
# or corrects it.
#
# 0.08 was the Franka Hand's spec (the gripper this repo originally
# loaded). REV20 P0.2 swapped the competition robot to a Robotiq 2F-85 --
# named for its 85mm stroke, the same 0.085m figure REV20's own session
# briefs already use ("far past a 2F-85's 0.085 m opening") -- but nothing
# updated this default, and no caller passes gripper_max_opening_m
# explicitly (2026-08-11 audit), so every grasp-point width check has been
# silently screening against the wrong gripper's spec since the swap.
GRIPPER_MAX_OPENING_M = 0.085

# World-Z clearance added above the mask's nearest (top) surface point,
# same convention as `perception_targets.bbox_grasp_target`'s
# `bbox_top_z + rim_clearance_m` -- added in WORLD Z, not camera depth,
# because a camera-depth offset is ambiguous for an oblique view angle.
DEFAULT_GRASP_CLEARANCE_M = 0.02


@dataclass
class GraspCandidate:
    object: str
    xyz: tuple[float, float, float]
    yaw_rad: float
    width_m: float
    width_ok: bool
    mask_pixel_count: int
    sides: list[IkScreenResult] = field(default_factory=list)

    @property
    def any_side_feasible(self) -> bool:
        return any(s.ik_feasible for s in self.sides)

    @property
    def best_feasible_norm_m(self) -> float | None:
        feasible = [
            s.target_norm_from_base_m
            for s in self.sides
            if s.ik_feasible and s.target_norm_from_base_m is not None
        ]
        return min(feasible) if feasible else None


def segment(
    seg_data: dict[str, Any], object_prim_substring: str
) -> np.ndarray:
    """Boolean HxW mask for every pixel whose instance id maps (via
    ``seg_data["info"]["idToLabels"]``) to a prim path containing
    ``object_prim_substring``. ``seg_data`` is the raw dict
    ``rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
    .get_data()`` returns -- ground-truth today; nothing downstream may
    assume that (REV16 C.0/C.1: the mask source is swappable).

    Returns an all-``False`` mask (not an exception) if no id matches --
    callers treat "object not visible" as a real perception outcome via
    ``grasp_point_from_mask`` returning ``None``, not via a crash.
    """
    data = np.asarray(seg_data["data"])
    id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
    matching_ids = {
        int(k)
        for k, v in id_to_labels.items()
        if object_prim_substring in str(v)
    }
    if not matching_ids:
        return np.zeros_like(data, dtype=bool)
    return np.isin(data, list(matching_ids))


def mask_centroid_px(mask: np.ndarray) -> tuple[float, float]:
    """(u, v) pixel centroid of a boolean mask. Raises on an empty mask --
    callers must treat "object not visible" as a real perception outcome,
    never silently grasp at (0, 0)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask -- object not visible in this frame")
    return float(xs.mean()), float(ys.mean())


def mask_principal_axes_px(
    mask: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """PCA on mask pixel coordinates. Returns ``(major_unit_xy,
    minor_unit_xy, major_extent_px, minor_extent_px)`` -- extents are the
    pixel span of the mask projected onto each axis (the object's
    apparent length/width in the image), not yet a physical unit (see
    ``meters_per_pixel``)."""
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        raise ValueError("mask too small for a principal-axis fit")
    pts = np.stack([xs, ys], axis=1).astype(float)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]
    proj_major = centered @ major
    proj_minor = centered @ minor
    major_extent = float(proj_major.max() - proj_major.min())
    minor_extent = float(proj_minor.max() - proj_minor.min())
    return (
        (float(major[0]), float(major[1])),
        (float(minor[0]), float(minor[1])),
        major_extent,
        minor_extent,
    )


def _as_4x4(matrix: Any) -> np.ndarray:
    """Accept either a nested 4x4 sequence or a flat length-16 sequence --
    `camera_params` annotators commonly return the latter."""
    arr = np.asarray(matrix, dtype=float).reshape(4, 4)
    return arr


def project_to_pixel(
    world_xyz: tuple[float, float, float],
    view_matrix: Any,
    proj_matrix: Any,
    width_px: int,
    height_px: int,
) -> tuple[float, float, float]:
    """World XYZ -> ``(u_px, v_px, depth_m)``. Same row-vector * matrix
    convention as ``SceneCamera.project``
    (``scripts/task3/probe_gemini_er_vs_ground_truth.py``), generalized
    from a USD ``Gf.Camera`` frustum to raw matrix arrays. Exists mainly
    as the round-trip check for ``project_to_world`` below."""
    view = _as_4x4(view_matrix)
    proj = _as_4x4(proj_matrix)
    world_pt = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1.0])
    cam_pt = world_pt @ view
    clip_pt = cam_pt @ proj
    w = clip_pt[3]
    if w == 0.0:
        raise ValueError("degenerate projection (w=0)")
    ndc_x = clip_pt[0] / w
    ndc_y = clip_pt[1] / w
    u_px = (ndc_x * 0.5 + 0.5) * width_px
    v_px = (1.0 - (ndc_y * 0.5 + 0.5)) * height_px
    depth_m = -cam_pt[2]
    return (float(u_px), float(v_px), float(depth_m))


def project_to_world(
    u_px: float,
    v_px: float,
    depth_m: float,
    view_matrix: Any,
    proj_matrix: Any,
    width_px: int,
    height_px: int,
) -> tuple[float, float, float]:
    """Inverse of ``project_to_pixel``: pixel + depth -> world XYZ.
    ``depth_m`` must be the straight-line distance along the camera's
    view axis (a ``distance_to_image_plane`` reading), NOT
    ``distance_to_camera`` -- same distinction
    ``sim_camera_perception.back_project``'s docstring already flags for
    the position-only pipeline; passing the wrong one here silently
    corrupts every grasp point the same way."""
    if depth_m <= 0.0:
        raise ValueError("depth_m must be positive")
    view = _as_4x4(view_matrix)
    proj = _as_4x4(proj_matrix)
    ndc_x = (u_px / width_px) * 2.0 - 1.0
    ndc_y = 1.0 - (v_px / height_px) * 2.0
    cam_x = ndc_x * depth_m / proj[0][0]
    cam_y = ndc_y * depth_m / proj[1][1]
    cam_z = -depth_m
    cam_pt = np.array([cam_x, cam_y, cam_z, 1.0])
    view_inv = np.linalg.inv(view)
    world_pt = cam_pt @ view_inv
    w = world_pt[3]
    if w == 0.0:
        raise ValueError("degenerate unprojection (w=0)")
    return (
        float(world_pt[0] / w),
        float(world_pt[1] / w),
        float(world_pt[2] / w),
    )


def meters_per_pixel(proj_matrix: Any, depth_m: float, width_px: int) -> float:
    """1 pixel of image-plane extent, at ``depth_m`` along the camera's
    view axis, in metres -- derived from the projection matrix's own
    ``[0][0]`` entry (row-vector * matrix convention: ``clip_x = cam_x *
    proj[0][0]``, and ``w = -cam_z = depth_m`` for a camera looking down
    -Z). This is the raw-matrix form of the same relation
    ``SceneCamera.meters_per_pixel_norm`` computes via the frustum
    window -- proven equal by the round-trip test, not re-derived from
    scratch."""
    if depth_m <= 0.0:
        raise ValueError("depth_m must be positive")
    proj = _as_4x4(proj_matrix)
    full_width_at_depth = 2.0 * depth_m / float(proj[0][0])
    return float(full_width_at_depth / width_px)


def grasp_point_from_mask(
    object_name: str,
    mask: np.ndarray,
    depth: np.ndarray,
    view_matrix: Any,
    proj_matrix: Any,
    width_px: int,
    height_px: int,
    *,
    clearance_m: float = DEFAULT_GRASP_CLEARANCE_M,
    gripper_max_opening_m: float = GRIPPER_MAX_OPENING_M,
) -> GraspCandidate | None:
    """Centroid + principal-axis grasp point.

    Back-projects the mask's pixel centroid at its NEAREST (minimum)
    depth to get the object's real top-surface world point, then adds
    ``clearance_m`` in WORLD Z (module docstring). Yaw is set
    perpendicular to the mask's major axis, so the gripper's closing
    axis crosses the object's MINOR axis -- a rim/handle grasp, not a
    centre pinch across the long axis. ``width_m``/``width_ok`` measure
    that minor-axis extent in real metres via ``meters_per_pixel`` at the
    surface depth and check it against ``gripper_max_opening_m``.

    Returns ``None`` (never raises) if the mask is empty, too small for a
    PCA fit, or has no valid depth reading -- callers MUST fall back to
    the existing constant-offset grasp path on ``None``, per REV16 C.2's
    "the night must never stall on an exception."
    """
    try:
        u, v = mask_centroid_px(mask)
        major_xy, _minor_xy, _major_extent, minor_extent_px = (
            mask_principal_axes_px(mask)
        )
    except ValueError:
        return None

    ys, xs = np.nonzero(mask)
    mask_depths = np.asarray(depth)[ys, xs]
    mask_depths = mask_depths[np.isfinite(mask_depths) & (mask_depths > 0)]
    if mask_depths.size == 0:
        return None
    surface_depth_m = float(mask_depths.min())

    surface_xyz = project_to_world(
        u, v, surface_depth_m, view_matrix, proj_matrix, width_px, height_px
    )
    grasp_xyz = (
        surface_xyz[0],
        surface_xyz[1],
        surface_xyz[2] + clearance_m,
    )

    # Perpendicular to the major (long) axis -- image-space angle,
    # matching perception_targets.py's er_grasp_targets_batched
    # grasp_angle_deg convention (gripper closing-axis rotation relative
    # to the image's horizontal axis).
    yaw_rad = math.atan2(major_xy[1], major_xy[0]) + math.pi / 2.0

    mpp = meters_per_pixel(proj_matrix, surface_depth_m, width_px)
    width_m = float(minor_extent_px * mpp)
    width_ok = bool(width_m <= gripper_max_opening_m)

    return GraspCandidate(
        object=object_name,
        xyz=tuple(round(c, 4) for c in grasp_xyz),
        yaw_rad=round(yaw_rad, 4),
        width_m=round(width_m, 4),
        width_ok=width_ok,
        mask_pixel_count=int(mask.sum()),
    )


def screen_grasp_candidate_ik(
    world: Any, candidate: GraspCandidate
) -> GraspCandidate:
    """Screen ``candidate`` through the SAME real IK dry-run
    ``perception_targets.screen_candidate_ik`` uses -- not reimplemented
    here, per this module's own docstring."""
    candidate.sides = screen_candidate_ik(
        world, candidate.xyz, candidate.yaw_rad
    )
    return candidate
