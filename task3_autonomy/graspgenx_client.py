# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""EBiM Task 3 (2026-08-20): live GraspGenX candidate generation.

GraspGenX (https://github.com/NVlabs/GraspGenX, Apache 2.0) runs as a
SEPARATE process in its own `uv`-managed venv, not imported into the Isaac
Sim Kit process -- this repo's own GOTCHAS.md is explicit that pinned
`warp`/`torch`/cuRobo versions inside the Isaac container are fragile to
unrelated imports, and GraspGenX brings its own independent torch/CUDA
stack. Communication is file-based (point cloud in, .npz out), the same
shape as every other external-process integration in this project.

This module owns exactly two things, both pure/testable without Isaac:
1. Building a GraspGenX-frame object point cloud from a captured
   depth+segmentation frame (reusing the SAME per-pixel unprojection math
   `task3_autonomy/perception_grasp.project_to_world` already uses and
   this project's own round-trip test already proves correct -- just
   vectorized across a masked region instead of one pixel).
2. Converting GraspGenX's returned camera-frame grasps to world XYZ+yaw
   (the identical axis_flip + fingertip-offset + two-point-yaw math
   `scripts/task3/graspgenx_world_attempt_live.py` already GPU-validated
   to 0.6-4cm accuracy -- reused here, not re-derived).

Object-specific position overrides (e.g. cup's body-squeeze pivot in
`graspgenx_world_attempt_live.py`, grounded in real mesh measurement) stay
in the calling script, not here -- this module is object-agnostic.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# GraspGenX's own camera convention (X-right, Y-down, Z-forward) vs. the
# USD/Isaac camera convention (X-right, Y-up, Z-backward, camera looks
# down -Z) differ by a fixed 180-degree rotation about X -- this exact
# matrix, applied in either direction (it is its own inverse), is what
# `graspgenx_world_attempt_live.py` already uses and GPU-validated.
AXIS_FLIP = np.diag([1.0, -1.0, -1.0])

# scripts/task3/graspgenx_world_attempt_live.py, GRASPGENX_FINGERTIP_DEPTH_M:
# GraspGenX encodes a gripper-BASE pose; the real contact point is this
# fixed offset along the grasp's own local +Z.
DEFAULT_FINGERTIP_DEPTH_M = 0.136


@dataclass
class GraspGenXCandidate:
    index: int
    confidence: float
    fingertip_world_xyz: tuple[float, float, float]
    yaw_rad: float


def object_point_cloud_camera_frame(
    depth: np.ndarray,
    mask: np.ndarray,
    proj_matrix: Any,
    width_px: int,
    height_px: int,
    *,
    max_points: int = 8000,
    rng_seed: int = 0,
) -> np.ndarray:
    """Masked depth pixels -> (N,3) point cloud in GraspGenX's own camera
    convention (X-right, Y-down, Z-forward).

    Vectorized form of the exact per-pixel formula
    ``task3_autonomy.perception_grasp.project_to_world`` uses to go from
    (u_px, v_px, depth_m) to camera-space XYZ (before that function's own
    additional step to WORLD space, which this does separately via the
    live base/camera pose -- see ``graspgenx_candidates_to_world`` below),
    with ``AXIS_FLIP`` applied to land in GraspGenX's convention instead
    of USD's.
    """
    if mask.shape != depth.shape:
        raise ValueError(f"mask shape {mask.shape} != depth shape {depth.shape}")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty mask -- object not visible in this frame")
    d = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    ys, xs, d = ys[valid], xs[valid], d[valid]
    if xs.size == 0:
        raise ValueError("mask has no pixels with valid (finite, positive) depth")

    proj = np.asarray(proj_matrix, dtype=np.float64).reshape(4, 4)
    ndc_x = (xs.astype(np.float64) / width_px) * 2.0 - 1.0
    ndc_y = 1.0 - (ys.astype(np.float64) / height_px) * 2.0
    cam_x = ndc_x * d / proj[0][0]
    cam_y = ndc_y * d / proj[1][1]
    cam_z = -d
    cam_pts = np.stack([cam_x, cam_y, cam_z], axis=1)
    cv_pts = cam_pts @ AXIS_FLIP.T

    if cv_pts.shape[0] > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(cv_pts.shape[0], size=max_points, replace=False)
        cv_pts = cv_pts[idx]
    return cv_pts.astype(np.float32)


def run_graspgenx_inference(
    pc_camera_frame: np.ndarray,
    *,
    graspgenx_root: Path,
    work_dir: Path,
    gripper_name: str = "robotiq_2f_85",
    topk_num_grasps: int = 20,
    timeout_s: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Invoke GraspGenX as a subprocess in its own venv. Returns
    (grasps (K,4,4) float32, confidences (K,) float32) in the SAME
    camera-frame convention `pc_camera_frame` was given in (GraspGenX
    translates its output back to the input frame before saving -- see
    `GraspGenX/scripts/ebim_infer_object_grasps.py`).

    Raises ``RuntimeError`` on any non-zero exit or malformed output --
    callers must treat that as "no candidates this cycle" (re-perceive /
    retry / abort per the existing GraspGenX design's retry rules), never
    silently substitute a stale or fabricated candidate.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    pc_path = work_dir / f"pc_{run_id}.npy"
    out_path = work_dir / f"grasps_{run_id}.npz"
    np.save(pc_path, pc_camera_frame.astype(np.float32))

    venv_python = graspgenx_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise RuntimeError(
            f"GraspGenX venv not found at {venv_python} -- run `uv sync` "
            f"in {graspgenx_root} first"
        )
    script = graspgenx_root / "scripts" / "ebim_infer_object_grasps.py"
    cmd = [
        str(venv_python),
        str(script),
        "--pc_npy",
        str(pc_path),
        "--gripper_name",
        gripper_name,
        "--out",
        str(out_path),
        "--topk_num_grasps",
        str(topk_num_grasps),
    ]
    # GPU-confirmed (2026-08-20): the calling process here is Isaac Sim's
    # own `/isaac-sim/python.sh`, which sets a PYTHONPATH loaded with
    # Isaac's OWN site-packages (including an old `huggingface_hub`
    # lacking `cached_download`). Inheriting that env into this
    # completely separate venv's subprocess shadowed GraspGenX's own
    # correct `huggingface_hub`/`diffusers` with Isaac's, breaking the
    # import outright -- same class of cross-contamination this repo's
    # own GOTCHAS.md documents for warp/curobo, now confirmed for
    # GraspGenX too. Strip PYTHON* so the subprocess sees only its own
    # venv, the same isolation `import warp` before cuRobo already relies
    # on for a different pair of libraries.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    result = subprocess.run(
        cmd,
        cwd=str(graspgenx_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    if result.returncode != 0 or "EBIM_INFER_OK" not in result.stdout:
        raise RuntimeError(
            f"GraspGenX inference failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}"
        )
    data = np.load(out_path)
    grasps = np.asarray(data["instance_1_grasps"], dtype=np.float32)
    confidences = np.asarray(data["instance_1_confidences"], dtype=np.float32)
    if grasps.shape[0] == 0 or grasps.shape[1:] != (4, 4):
        raise RuntimeError(f"malformed GraspGenX output: grasps.shape={grasps.shape}")
    return grasps, confidences


def graspgenx_candidates_to_world(
    grasps_camera_frame: np.ndarray,
    confidences: np.ndarray,
    cam_pos_w: np.ndarray,
    cam_quat_w: np.ndarray,
    *,
    quat_rotate: Any,
    fingertip_depth_m: float = DEFAULT_FINGERTIP_DEPTH_M,
) -> list[GraspGenXCandidate]:
    """GraspGenX camera-frame grasps -> world XYZ + yaw, confidence-sorted
    descending. Identical math to
    `scripts/task3/graspgenx_world_attempt_live.py` (fingertip offset
    along local Z, a second point along local X for yaw recovery,
    AXIS_FLIP to USD camera convention, then the live camera pose) --
    `quat_rotate` is passed in (`task3_autonomy.curobo_grasp_planner
    .quat_rotate`) rather than imported here so this module stays
    torch-free and importable/unit-testable on CPU without Isaac.
    """
    import torch

    cam_pos_w_t = torch.as_tensor(cam_pos_w, dtype=torch.float64)
    cam_quat_w_t = torch.as_tensor(cam_quat_w, dtype=torch.float64)

    def _to_world(p_usdcam: np.ndarray) -> np.ndarray:
        v = torch.as_tensor(p_usdcam, dtype=torch.float64)
        world_v = cam_pos_w_t + quat_rotate(
            cam_quat_w_t.unsqueeze(0), v.unsqueeze(0)
        ).squeeze(0)
        return world_v.numpy()

    out: list[GraspGenXCandidate] = []
    for i in range(grasps_camera_frame.shape[0]):
        G = grasps_camera_frame[i]
        R_gg = G[:3, :3]
        t_gg = G[:3, 3]
        fingertip_gg = t_gg + fingertip_depth_m * R_gg[:, 2]
        x_tip_gg = fingertip_gg + R_gg[:, 0]
        fingertip_usdcam = AXIS_FLIP @ fingertip_gg
        x_tip_usdcam = AXIS_FLIP @ x_tip_gg
        t_grasp_w = _to_world(fingertip_usdcam)
        x_tip_w = _to_world(x_tip_usdcam)
        x_dir_w = x_tip_w - t_grasp_w
        yaw_rad = float(np.arctan2(x_dir_w[1], x_dir_w[0]))
        out.append(
            GraspGenXCandidate(
                index=i,
                confidence=float(confidences[i]),
                fingertip_world_xyz=(
                    float(t_grasp_w[0]),
                    float(t_grasp_w[1]),
                    float(t_grasp_w[2]),
                ),
                yaw_rad=yaw_rad,
            )
        )
    out.sort(key=lambda c: c.confidence, reverse=True)
    return out
