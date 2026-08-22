# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""N2 (plans/OVERNIGHT_VM_B_2026-08-03.md) -- perception-driven adaptive
grasp targets, replacing the single hardcoded ``z_offset = 0.075`` /
``PREGRASP_Z = 1.05`` every Stage-4 object shares today
(``world_isaac.py:748-757``).

Default-off, new module: nothing in the live grasp path imports this yet
(per the plan's own "wire into the live grasp path only after it wins
offline" rule, carried over from L2/``er_grasp_pose.py``). Two candidate
sources per object, ranked ER-first:

1. **ER** -- Gemini Robotics-ER 2, ONE batched call for every object
   (position + orientation), unprojected with
   ``probe_gemini_er_vs_ground_truth.SceneCamera`` (the same
   self-checked camera math B3b measured at 5.71cm mean error against
   ground truth), anchored on each object's own ground-truth DEPTH so
   ER only ever supplies "where on the object", never "where the
   object is".
2. **BBOX** -- geometric fallback with no API: the object's own real
   USD world-aligned bounding box (top for rim height, live centre for
   XY), so a rate-limited/exhausted ER key still produces an
   object-specific, non-hardcoded target instead of stalling or
   reverting to the shared constant.

Every candidate is screened through a REAL kinematic IK dry-run
(``scripts/common/dual_arm_lula.DualArmLulaIK.solve``, the exact solver
``DualArmController.command()`` uses every tick in production, called
here with no physical ticking and no base motion -- a pure feasibility
query) against the robot's actual current base pose, for BOTH arms, so
"IK-feasible" here means the same solver the live grasp path trusts,
not a re-derived approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

REACH_LIMIT_M = 0.855


@dataclass
class IkScreenResult:
    side: str
    ik_feasible: bool
    target_norm_from_base_m: float | None


@dataclass
class ScreenedCandidate:
    object: str
    source: str  # "er" | "bbox"
    rank: int
    xyz: tuple[float, float, float]
    yaw_rad: float
    er_label: str | None = None
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


def bbox_grasp_target(
    world: Any, name: str, *, rim_clearance_m: float = 0.02
) -> tuple[float, float, float] | None:
    """Geometric fallback grasp target: live XY centre, Z anchored on the
    object's OWN real USD world-aligned bbox top (not a shared constant).
    Returns None if the object's prim/bbox cannot be resolved -- callers
    must treat that as "this source is unavailable for this object", not
    silently fall through to a hardcoded value.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    view = world.object_views.get(name)
    if view is None:
        return None
    prim_paths = getattr(view, "prim_paths", None)
    if not prim_paths:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_paths[0])
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    bbox_top_z = float(rng.GetMax()[2])
    live_x, live_y, _ = world.object_position(name)
    return (live_x, live_y, bbox_top_z + rim_clearance_m)


def er_grasp_targets_batched(
    frame_path: Any,
    objects: list[str],
    api_keys: list[str],
    call_er_fn: Any,
) -> tuple[dict[str, dict], float, str]:
    """ONE compound ER call asking for a grasp point + orientation for
    every object at once (cost discipline: max information per call, not
    max calls). Returns (by_object_name, latency_s, key_suffix)."""
    prompt = (
        "For EACH of these objects, point to the single best grasp point "
        "for a parallel-jaw gripper to pick it up (e.g. a handle or rim "
        "if the object has one, not necessarily its geometric centre): "
        + ", ".join(objects)
        + '. Return a JSON array with one entry per object: {"object": '
        '<one of the names above, exactly>, "point": [y, x], '
        '"grasp_angle_deg": <float 0-180, rotation of the gripper\'s '
        "closing axis relative to the image's horizontal axis that "
        "would let the jaws close around the object at that point "
        'without hitting it edge-on>}. "point" is normalized 0-1000, '
        "[y, x] order."
    )
    raw, latency, key_suffix = call_er_fn(frame_path, prompt, api_keys)
    by_name: dict[str, dict] = {}
    if isinstance(raw, list):
        for entry in raw:
            name = entry.get("object")
            if name in objects and name not in by_name:
                by_name[name] = entry
    return by_name, latency, key_suffix


def screen_candidate_ik(
    world: Any, xyz: tuple[float, float, float], yaw_rad: float
) -> list[IkScreenResult]:
    """Pure kinematic IK dry-run for `xyz`/`yaw_rad` against the robot's
    ACTUAL current base pose, for both arms. No ticking, no base motion --
    `DualArmLulaIK.solve` (scripts/common/dual_arm_lula.py) is the same
    solver `DualArmController.command()` calls every production tick."""
    arms = world.arms
    quat = world._m["_quaternion_from_rpy"](math.pi, 0.0, yaw_rad)
    root_position, root_orientation = arms._root_pose(world.robot)
    result = arms._ik.solve(
        xyz,
        xyz,
        quat,
        quat,
        spine_position=arms.spine,
        base_position=root_position,
        base_orientation_wxyz=root_orientation,
    )
    out = []
    for side, succeeded in (
        ("left", result.left_succeeded),
        ("right", result.right_succeeded),
    ):
        rel = world._arm_base_relative(side, xyz)
        norm_m = rel[1] if rel else None
        out.append(
            IkScreenResult(
                side=side,
                ik_feasible=bool(succeeded),
                target_norm_from_base_m=norm_m,
            )
        )
    return out


def build_ranked_candidates(
    world: Any,
    name: str,
    er_entry: dict | None,
    cam: Any,
    ground_truth_depth_m: float,
) -> list[ScreenedCandidate]:
    """Rank 0: ER (if ER returned a usable point for this object).
    Rank 1: BBOX (always attempted -- the "night cannot stall" fallback).
    Each is screened through real IK before being returned; callers pick
    the first `any_side_feasible` candidate, falling down the list on
    rejection, per the plan's own "do not try one pose and give up" rule.
    """
    candidates: list[ScreenedCandidate] = []
    rank = 0

    if er_entry is not None:
        point = er_entry.get("point")
        angle_deg = er_entry.get("grasp_angle_deg")
        if point and len(point) == 2:
            py, px = float(point[0]), float(point[1])
            # Anchor depth on the OBJECT'S OWN ground-truth depth (B3b/L2
            # discipline) -- ER supplies where-on-the-object, never
            # where-the-object-is.
            er_xyz = cam.unproject_at_depth(py, px, ground_truth_depth_m)
            yaw_rad = (
                math.radians(float(angle_deg))
                if angle_deg is not None
                else 0.0
            )
            cand = ScreenedCandidate(
                object=name,
                source="er",
                rank=rank,
                xyz=tuple(round(v, 4) for v in er_xyz),
                yaw_rad=round(yaw_rad, 4),
                er_label=er_entry.get("label"),
            )
            cand.sides = screen_candidate_ik(world, cand.xyz, cand.yaw_rad)
            candidates.append(cand)
            rank += 1

    bbox_xyz = bbox_grasp_target(world, name)
    if bbox_xyz is not None:
        cand = ScreenedCandidate(
            object=name, source="bbox", rank=rank, xyz=bbox_xyz, yaw_rad=0.0
        )
        cand.sides = screen_candidate_ik(world, cand.xyz, cand.yaw_rad)
        candidates.append(cand)
        rank += 1

    return candidates


def best_feasible(
    candidates: list[ScreenedCandidate],
) -> ScreenedCandidate | None:
    """First candidate (by rank) that is IK-feasible on at least one arm --
    "fall down the ranked list on rejection", never "try one and stop"."""
    for cand in candidates:
        if cand.any_side_feasible:
            return cand
    return None
