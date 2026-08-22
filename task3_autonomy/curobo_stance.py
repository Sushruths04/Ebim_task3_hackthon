# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""GATE B0 (handoff sec 68): the cuRobo batch-IK stance search, PORTED
into the production path.

sec 64 (GATE C2.5A) proved -- 8/8 successes vs 9/9 failures, 100%
correlation, zero ambiguous cases -- that `stance_for()`'s FIXED
STANCE_REACH_RADIUS_M (~0.866m, task3_autonomy/navigation.py) is the root
cause of most Stage 1/4 reach failures: it nets a reachable arm-frame
distance for some objects and an unreachable one for others, purely
depending on which direction _rotate_to_clear_island() had to rotate.

sec 65-67 (GATE C2.5B/C3/C4) proved the fix: sample many candidate mobile-
base stances on an annulus around the object, batch-solve cuRobo IK for
ALL of them in one solve_pose() call, reject any whose footprint overlaps
the kitchen island, and pick the closest practically-reachable candidate
to the robot's current position (not the largest cuRobo margin -- sec 67's
own T5 run1 finding: the largest-margin candidate can be geometrically
valid but operationally unreachable by navigate_to()'s fixed budget if it
lands on the far side of the island from the robot's actual position).

Until this file existed, that fix lived ONLY in
scripts/task3/curobo/probe_grasp_lift_c4.py -- a probe script, never
called by task3_pipeline or task3_autonomy.navigation (REVIEW #8, handoff
sec 68). Every scoring run was still using the broken fixed radius. This
module is that same logic, refactored from the probe's
_make_curobo_stance_for() closure into a reusable class so
task3_pipeline.world_isaac.IsaacWorld._stance_for can call it directly.

Built lazily, once per episode (see IsaacWorld._stance_for) -- it needs
live robot body poses, which only exist after IsaacWorld.reset().
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from task3_autonomy.navigation import (
    BASE_HALF_WIDTH,
    ISLAND_DETOUR_MARGIN_M,
    KITCHEN_ISLAND_BBOX,
    point_clears_island,
    route_around_island,
)
from task3_autonomy.navigation import stance_for as _fallback_stance_for

# T4 follow-up (plans/LOOP_PROMPT_VM_A_REV5.md): the live GPU verification
# of route_around_island (2026-08-04, plans/SYNC.md) surfaced a REAL
# regression -- stalled navigate_to count went 2 -> 9 on an otherwise
# identical run. Traced to real base positions in that log sitting at
# x~-4.90 to -4.91, i.e. AT `point_clears_island`'s own BASE_HALF_WIDTH-only
# boundary (x_min-half_width=-4.91) with millimeter-scale real clearance --
# not caused by the new detour route itself, but by candidate generation
# below never having had a safety margin: `point_clears_island` here only
# ever checked bare BASE_HALF_WIDTH, so a candidate 1cm past the raw
# threshold was accepted and could actually get chosen and driven to once
# route_around_island stopped rejecting every reachable-in-principle
# candidate outright. Reusing the same margin the detour math already
# uses, rather than inventing a second constant.
_CANDIDATE_ISLAND_MARGIN_M = ISLAND_DETOUR_MARGIN_M

REPO_ROOT = Path(__file__).resolve().parents[1]
CUROBO_ROBOT_YML = (
    REPO_ROOT / "scripts" / "task3" / "curobo" / "fr3_duo_right_arm.yml"
)
# BLOCKER 3 (handoff sec 51/58): warp must already be import-able (and, for
# an in-process caller that has not already done so, imported) before any
# cuRobo import touches it, or the import chain hangs. IsaacWorld's caller
# (run_task3.py) front-loads `sys.path.insert + import warp` before
# AppLauncher for exactly this reason when curobo stance search is enabled.
CUROBO_SPIKE_PATH = "/workspace/curobo_spike"

# sec 65: this project's standing num_seeds default (32, used everywhere
# else in this workspace region) silently under-converges for base-stance
# IK targets -- 128 gave a 24x error reduction (0.419m -> 0.017m) on an
# otherwise-identical solve. sec 66: cuRobo's own default position_tolerance
# (0.005m) is tighter than the ~0.017m floor imposed by this project's
# existing both-tool-frames-to-one-point approximation (half the measured
# 0.034m real finger separation) -- 0.02m matches production reach()'s own
# DEFAULT_POSITION_TOLERANCE_M (task3_autonomy/arms.py:27), the same
# practical standard the rest of the codebase already uses.
NUM_SEEDS = 128
POSITION_TOLERANCE_M = 0.02

# Candidate annulus: 9 radii x 35 yaw steps (10 deg apart) + the old
# stance_for() candidate at index 0 -- same grid probe_stance_search_c25b.py
# validated (handoff sec 65).
_RADII_MIN_M = 0.45
# S2 sub-fix 2 (plans/LOOP_PROMPT_VM_A.md rev 2, P4): S1 proved the outer
# annulus edge sat right at the ~0.855m singularity boundary (chosen_radius_m
# == min_accepted_radius_m == 0.85 on every successful search, GATE S1).
# 0.85 leaves only ~5mm of margin against that ceiling -- capping the annulus
# itself to 0.78m keeps every candidate this search can ever propose at least
# ~75mm inside the reach limit, on top of (not a replacement for) sub-fix 1's
# ranking-time reach-margin term.
_RADII_MAX_M = 0.78
_N_RADII = 9
_N_YAW_STEPS = 36  # linspace(0, 2*pi, 36)[:-1] -> 35 steps, 10 deg apart
# Generous upper bound on total candidates (9 * 35 + 1 = 316) so the IK
# solver's batch buffers are sized once, in __init__, not rebuilt per call.
MAX_CANDIDATES = 320


def _quat_mul(q1, q2):
    """Batched Hamilton product, wxyz convention. q1, q2: (N, 4)."""
    import torch

    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def _quat_conj(q):
    import torch

    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def _quat_rotate(q, v):
    """Batched rotate v (N,3) by q (N,4), wxyz."""
    import torch

    w = q[..., 0]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)


def _yaw_quat(yaw):
    """Batched yaw-only quaternion (N,), wxyz, rotation about +Z."""
    import torch

    half = yaw * 0.5
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(w)
    return torch.stack([w, zeros, zeros, z], dim=-1)


# T4 (plans/LOOP_PROMPT_VM_A_REV5.md): this used to be a local straight-
# line-only `_segment_clears_island`, checked once and only accepted if
# that single straight leg cleared. Moved to task3_autonomy/navigation.py
# (this file's own docstring already says stance-search geometry lives
# there; the straight-line check plus the new `route_around_island` U-shaped
# detour both belong with `KITCHEN_ISLAND_BBOX`/`point_clears_island`, not
# duplicated here). A candidate is now path-clear if EITHER the straight
# leg clears OR a detour around the island does --
# `NavigateTo` (task3_autonomy/skills.py) drives that same detour via
# `insert_island_detours`, so accept and actually-drivable stay consistent
# (the historical stall this check exists to prevent, handoff sec 69,
# was accepting an UNDRIVABLE straight line -- this fix keeps rejecting
# those, it only additionally accepts ones a real detour route can drive).
def _segment_clears_island(
    p0: tuple[float, float],
    p1: tuple[float, float],
    half_width: float = BASE_HALF_WIDTH,
    bbox: dict = KITCHEN_ISLAND_BBOX,
    obstacles: list[dict] | None = None,
) -> bool:
    return (
        route_around_island(p0, p1, half_width, bbox, obstacles=obstacles)
        is not None
    )


class CuroboStanceSearch:
    """Batch-IK stance search built once per episode from a live IsaacWorld.

    Call .stance_for(object_xy, approach) to get the best (xy, yaw) stance,
    or None if no candidate is practically reachable -- the caller
    (IsaacWorld._stance_for) falls back to navigation.stance_for() in
    that case, per GATE B0's instruction to never let a search miss abort
    a stage.
    """

    def __init__(
        self,
        world: Any,
        device: str = "cuda:0",
        rate_both_arms: bool = False,
    ) -> None:
        if CUROBO_SPIKE_PATH not in sys.path:
            sys.path.insert(0, CUROBO_SPIKE_PATH)
        import torch
        import warp as wp  # noqa: F401 -- BLOCKER 3, must precede cuRobo import

        from curobo.inverse_kinematics import (
            InverseKinematics,
            InverseKinematicsCfg,
        )

        self.world = world
        self._device = device
        self._torch = torch

        robot = world.robot
        body_names = list(robot.body_names)
        root_idx = body_names.index("base")

        root_pos_w = robot.data.body_pos_w[0, root_idx].clone()
        root_quat_w = robot.data.body_quat_w[0, root_idx].clone()
        self._root_pos_w = root_pos_w
        root_quat_conj = _quat_conj(root_quat_w.unsqueeze(0))[0]

        self._offset_pos_local = {}
        self._offset_quat_local = {}
        for side, body_name in (
            ("right", "right_base"),
            ("left", "left_base"),
        ):
            arm_idx = body_names.index(body_name)
            arm_pos_w = robot.data.body_pos_w[0, arm_idx].clone()
            arm_quat_w = robot.data.body_quat_w[0, arm_idx].clone()
            self._offset_pos_local[side] = _quat_rotate(
                root_quat_conj.unsqueeze(0),
                (arm_pos_w - root_pos_w).unsqueeze(0),
            )[0]
            self._offset_quat_local[side] = _quat_mul(
                root_quat_conj.unsqueeze(0), arm_quat_w.unsqueeze(0)
            )[0]
        self.rate_both_arms = rate_both_arms

        base_pose = world.adapter.pose()
        predicted_root_quat = _yaw_quat(
            torch.tensor([base_pose.yaw], device=device, dtype=torch.float32)
        )[0]
        self._yaw_quat_correction = _quat_mul(
            _quat_conj(predicted_root_quat.unsqueeze(0)),
            root_quat_w.unsqueeze(0),
        )[0]

        ik_config = InverseKinematicsCfg.create(
            robot=str(CUROBO_ROBOT_YML),
            num_seeds=NUM_SEEDS,
            self_collision_check=False,
            load_collision_spheres=False,
            max_batch_size=MAX_CANDIDATES,
            position_tolerance=POSITION_TOLERANCE_M,
        )
        self._ik = InverseKinematics(ik_config)
        print(
            f"PROGRESS: CuroboStanceSearch built, tool_frames="
            f"{self._ik.tool_frames}",
            flush=True,
        )

    def _arm_base_pose_for(self, xc, yc, yawc, side: str = "right"):
        torch = self._torch
        n = xc.shape[0]
        cand_root_pos = torch.stack(
            [xc, yc, self._root_pos_w[2].expand(n)], dim=-1
        )
        cand_root_quat = _quat_mul(
            _yaw_quat(yawc),
            self._yaw_quat_correction.unsqueeze(0).expand(n, 4),
        )
        offset_pos = self._offset_pos_local[side]
        offset_quat = self._offset_quat_local[side]
        arm_pos = cand_root_pos + _quat_rotate(
            cand_root_quat, offset_pos.unsqueeze(0).expand(n, 3)
        )
        arm_quat = _quat_mul(
            cand_root_quat, offset_quat.unsqueeze(0).expand(n, 4)
        )
        return arm_pos, arm_quat

    def _right_base_pose_for(self, xc, yc, yawc):
        # Kept as a thin alias -- pre-Q5 call sites/tests reference this
        # name directly; the real logic now lives in _arm_base_pose_for
        # so it can be reused for either arm.
        return self._arm_base_pose_for(xc, yc, yawc, side="right")

    def stance_for(
        self,
        object_xy: tuple[float, float],
        approach: str,
        contact_z: float | None = None,
    ):
        """Returns (xy, yaw) for the closest practically-reachable
        candidate, or None if none of the searched candidates converge --
        the caller must fall back to navigation.stance_for() in that case.

        N2 (handoff sec 93/94, GATE B1): `contact_z`, when given, adds a
        second batch-IK goal at that height and requires a candidate to
        be practically reachable at BOTH `vgl.PREGRASP_Z` and `contact_z`
        before it is accepted. Without this, a stance was only ever
        proven reachable at PREGRASP_Z (1.05 m) while `_push_object_to`
        descends to `contact_z` (~0.78 m, 0.27 m lower) -- a height this
        search never checked. Reuses the same candidate batch and IK
        solver already built for the PREGRASP_Z goal; this is a second
        goal, not a new solver.
        """
        torch = self._torch
        device = self._device
        vgl = self.world._m["vgl"]

        target_world = torch.tensor(
            [object_xy[0], object_xy[1], vgl.PREGRASP_Z],
            device=device,
            dtype=torch.float32,
        )
        top_down = torch.tensor(
            [0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32
        )

        old_xy, old_yaw = _fallback_stance_for(object_xy, approach)
        radii = torch.linspace(_RADII_MIN_M, _RADII_MAX_M, _N_RADII)
        yaws_theta = torch.linspace(
            0.0, 2.0 * math.pi, _N_YAW_STEPS, dtype=torch.float32
        )[:-1]
        cand_xy: list[tuple[float, float]] = []
        cand_yaw: list[float] = []
        for r in radii.tolist():
            for theta in yaws_theta.tolist():
                cx = object_xy[0] + r * math.cos(theta)
                cy = object_xy[1] + r * math.sin(theta)
                if not point_clears_island(
                    (cx, cy),
                    BASE_HALF_WIDTH + _CANDIDATE_ISLAND_MARGIN_M,
                    KITCHEN_ISLAND_BBOX,
                ):
                    continue
                cand_xy.append((cx, cy))
                cand_yaw.append(theta + math.pi)
        all_xy = [old_xy] + cand_xy
        all_yaw = [old_yaw] + cand_yaw
        n_total = len(all_xy)

        xc = torch.tensor(
            [p[0] for p in all_xy], device=device, dtype=torch.float32
        )
        yc = torch.tensor(
            [p[1] for p in all_xy], device=device, dtype=torch.float32
        )
        yawc = torch.tensor(all_yaw, device=device, dtype=torch.float32)
        from curobo.types import GoalToolPose, Pose

        quat_batch = top_down.unsqueeze(0).expand(n_total, 4).contiguous()

        def _solve_for_side(side: str):
            arm_pos_s, arm_quat_s = self._arm_base_pose_for(
                xc, yc, yawc, side=side
            )
            delta_s = target_world.unsqueeze(0).expand(n_total, 3) - arm_pos_s
            target_rel_s = _quat_rotate(_quat_conj(arm_quat_s), delta_s)
            goal_pose_s = Pose(position=target_rel_s, quaternion=quat_batch)
            result_s = self._ik.solve_pose(
                GoalToolPose.from_poses(
                    {name: goal_pose_s for name in self._ik.tool_frames},
                    num_goalset=1,
                )
            )
            pos_err_s = result_s.position_error.reshape(n_total).tolist()
            ok_s = [pe < POSITION_TOLERANCE_M for pe in pos_err_s]
            if contact_z is not None:
                target_world_contact = torch.tensor(
                    [object_xy[0], object_xy[1], contact_z],
                    device=device,
                    dtype=torch.float32,
                )
                delta_contact_s = (
                    target_world_contact.unsqueeze(0).expand(n_total, 3)
                    - arm_pos_s
                )
                target_rel_contact_s = _quat_rotate(
                    _quat_conj(arm_quat_s), delta_contact_s
                )
                goal_pose_contact_s = Pose(
                    position=target_rel_contact_s, quaternion=quat_batch
                )
                result_contact_s = self._ik.solve_pose(
                    GoalToolPose.from_poses(
                        {
                            name: goal_pose_contact_s
                            for name in self._ik.tool_frames
                        },
                        num_goalset=1,
                    )
                )
                pos_err_contact_s = result_contact_s.position_error.reshape(
                    n_total
                ).tolist()
                ok_s = [
                    ok and pe_c < POSITION_TOLERANCE_M
                    for ok, pe_c in zip(ok_s, pos_err_contact_s)
                ]
            return pos_err_s, ok_s

        # Q5 (SYNC 21/25, curobo_stance.py originally :313): every
        # candidate used to be rated against the right arm's base only.
        # `pos_err`/`practical_ok` below stay the RIGHT arm's own numbers
        # (unchanged default behaviour, and what the existing debug prints
        # below already report) -- rate_both_arms additionally OR's in the
        # left arm's feasibility per candidate, so a candidate unreachable
        # by the right arm but reachable by the left is no longer silently
        # discarded.
        pos_err, practical_ok = _solve_for_side("right")
        if self.rate_both_arms:
            pos_err_left, practical_ok_left = _solve_for_side("left")
            flipped = sum(
                1
                for r, l_ok in zip(practical_ok, practical_ok_left)
                if l_ok and not r
            )
            print(
                f"PROGRESS: curobo_stance_for({object_xy}) rate_both_arms: "
                f"{flipped}/{n_total - 1} candidates reachable by the LEFT "
                "arm only (right-only would have rejected them)",
                flush=True,
            )
            practical_ok = [
                r or l_ok for r, l_ok in zip(practical_ok, practical_ok_left)
            ]

        # sec T5 run1 (handoff sec 67): prefer the CLOSEST practical_ok
        # candidate to the robot's live current position, not the largest
        # cuRobo margin -- a kinematically-perfect stance on the far side
        # of the island from the robot's actual position stranded
        # navigate_to()'s fixed 25s/0.25mps budget part-way, failing the
        # whole chain even though the stance itself was valid.
        #
        # sec 69 follow-up: "closest" alone is not enough -- navigate_to()
        # drives straight at its target with no obstacle routing, so a
        # candidate whose straight-line path from the CURRENT position
        # cuts through the island stalls every time it is retried, even
        # though both endpoints individually clear the island footprint.
        # Reject those too (_segment_clears_island), not just rank by
        # distance.
        #
        # S2 sub-fix 1 (plans/SPRINT_30H_2026-08-02.md sec 4, S2): S1
        # proved (both by norm-clustering and, live, by
        # chosen_radius_m == min_accepted_radius_m == _RADII_MAX_M on
        # every successful search this sprint's s1_gpu_confirm_r2 run)
        # that ranking by `dist_from_current` alone has no reach-margin
        # term and systematically lands on the outer annulus edge, right
        # at the ~0.855m singularity boundary. Add a margin term:
        # score = dist_from_current + REACH_MARGIN_K * radius_from_object,
        # so a smaller (safer) radius is preferred even at some drive-
        # distance cost. REACH_MARGIN_K=1.0 is the exact break-even point
        # the brief specifies ("pick k so a 0.1m radius reduction
        # outweighs a 0.1m longer drive") -- both terms are already in
        # meters, so equal weighting is the natural, non-arbitrary choice
        # (Iron Rule 8: not a tuned magic number, the units justify it
        # directly).
        live_pose = self.world.adapter.pose()
        live_xy = (live_pose.x, live_pose.y)
        best_idx, best_dist, n_path_blocked = _select_closest_reachable(
            n_total, practical_ok, all_xy, live_xy, object_xy, REACH_MARGIN_K
        )

        if best_idx is None:
            print(
                f"PROGRESS: curobo_stance_for({object_xy}) found NO "
                f"practical-ok, path-clear candidate out of "
                f"{n_total - 1} searched ({n_path_blocked} rejected for "
                f"an island-crossing path from {live_xy}; old stance "
                f"pos_err={pos_err[0]:.5f}); falling back to "
                "navigation.stance_for()",
                flush=True,
            )
            return None

        # S1 (plans/SPRINT_30H_2026-08-02.md sec 1): purely additive --
        # log the chosen candidate's annulus radius (distance from the
        # OBJECT, not the robot) alongside the smallest radius among the
        # candidates this call actually accepted as practical_ok +
        # path-clear, to test the singularity hypothesis directly: does
        # "closest to the robot" (the only ranking term above) end up
        # picking radii near _RADII_MAX_M (0.85, at the ~0.855 reach
        # ceiling) even when a smaller-radius, more-margin candidate was
        # also on the table?
        chosen_radius = math.hypot(
            all_xy[best_idx][0] - object_xy[0],
            all_xy[best_idx][1] - object_xy[1],
        )
        accepted_radii = [
            math.hypot(
                all_xy[i][0] - object_xy[0], all_xy[i][1] - object_xy[1]
            )
            for i in range(1, n_total)
            if practical_ok[i] and _segment_clears_island(live_xy, all_xy[i])
        ]
        min_accepted_radius = min(accepted_radii) if accepted_radii else None
        min_radius_str = (
            "n/a"
            if min_accepted_radius is None
            else f"{min_accepted_radius:.4f}"
        )
        print(
            f"PROGRESS: curobo_stance_for({object_xy}) chose idx={best_idx} "
            f"xy={all_xy[best_idx]} yaw={all_yaw[best_idx]:.4f} "
            f"pos_err={pos_err[best_idx]:.5f} "
            f"dist_from_current={best_dist:.4f} "
            f"chosen_radius_m={chosen_radius:.4f} "
            f"min_accepted_radius_m={min_radius_str} "
            f"(old stance_for() would have been xy={old_xy} "
            f"pos_err={pos_err[0]:.5f})",
            flush=True,
        )
        return all_xy[best_idx], all_yaw[best_idx]


# S2 sub-fix 1: break-even weight between drive distance (m) and annulus
# radius (m) -- see the call site's comment for the derivation.
REACH_MARGIN_K = 1.0


def _select_closest_reachable(
    n_total: int,
    practical_ok: list[bool],
    all_xy: list[tuple[float, float]],
    live_xy: tuple[float, float],
    object_xy: tuple[float, float],
    margin_k: float = REACH_MARGIN_K,
    obstacles: list[dict] | None = None,
) -> tuple[int | None, float | None, int]:
    """Pure selection logic, extracted from `stance_for()` so it is
    CPU-testable without cuRobo/torch/GPU (S1,
    plans/SPRINT_30H_2026-08-02.md sec 1 step 3). Picks the practical_ok,
    path-clear candidate minimizing `dist_from_current +
    margin_k * radius_from_object` (S2 sub-fix 1: a reach-margin term,
    not pure closest-to-robot) -- index 0 (the navigation.py fallback
    candidate) is never considered, matching the original inline loop's
    `range(1, n_total)`.

    Returns (best_idx, best_dist, n_path_blocked); best_idx is None if no
    candidate is both practical_ok and path-clear. `best_dist` is the
    winning candidate's plain distance-from-robot (for telemetry), not
    its margin-adjusted score.
    """
    best_idx = None
    best_dist = None
    best_score = None
    n_path_blocked = 0
    for i in range(1, n_total):
        if not practical_ok[i]:
            continue
        if not _segment_clears_island(live_xy, all_xy[i], obstacles=obstacles):
            n_path_blocked += 1
            continue
        dist = math.hypot(all_xy[i][0] - live_xy[0], all_xy[i][1] - live_xy[1])
        radius = math.hypot(
            all_xy[i][0] - object_xy[0], all_xy[i][1] - object_xy[1]
        )
        score = dist + margin_k * radius
        if best_score is None or score < best_score:
            best_score = score
            best_dist = dist
            best_idx = i
    return best_idx, best_dist, n_path_blocked
