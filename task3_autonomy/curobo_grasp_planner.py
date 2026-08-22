# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Plan the grasp with cuMotion instead of servoing at it.

WHY THIS EXISTS, in one measurement. ER-2 puts the grasp point 0.048 m from
the object; the finger pads finish 0.132-0.165 m away. `recenter` servos for
800 ticks with IK solving on every one of them and still ends 0.05-0.07 m
from its own target against a 0.015 m tolerance. The arm misses a target it
already knows, so the bottleneck is trajectory EXECUTION, not perception and
not planning-where-to-go.

`curobo.motion_planner.MotionPlanner.plan_grasp()` returns approach, grasp
and lift trajectories -- the same three legs `world_isaac.reach()` drives by
hand -- as feasible, jerk-limited joint paths that END at the IK solution
rather than chasing it. cuRobo has been installed at
``/workspace/curobo_spike`` the whole time and this repo used only its
inverse_kinematics; see `scripts/task3/curobo/probe_motion_planner.py` for
the run that proved plan_grasp works here (approach/grasp/lift all True,
1000 waypoints each).

THE FRAME AND THE PADS, the two things that make this non-obvious:

* Goals are expressed in the robot YAML's ``base_link`` -- ``left_base`` for
  ``fr3_duo_left_arm.yml`` -- not in world. The probe's FK put the pads at
  ~(0.18, 0.0, 0.52), which is only sensible in that frame.
* The planner carries TWO tool frames, both finger pads, and requires a pose
  for each. They must STRADDLE the grasp point; giving both the same point
  asks for an impossible configuration and returns "Goalset planning
  returned None". That straddle is the same quantity `close` already
  measures as ``pad_midpoint_to_grasp_point_m``.
"""

from __future__ import annotations

import math
from typing import Any

CUROBO_SPIKE_PATH = "/workspace/curobo_spike"

# Half the jaw opening used to place the two pad goals either side of the
# grasp point. DERIVED, not chosen: half of
# `perception_grasp.GRIPPER_MAX_OPENING_M` (the 2F-85's 0.085 m), so the
# planned pads start exactly as far apart as the gripper can actually open.
DEFAULT_HALF_OPEN_M = 0.0425


def quat_rotate(q: Any, v: Any) -> Any:
    """Rotate vector(s) ``v`` by quaternion(s) ``q`` (wxyz), batched."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * _cross(xyz, v)
    return v + w * t + _cross(xyz, t)


def _cross(a: Any, b: Any) -> Any:
    return a.cross(b, dim=-1) if hasattr(a, "cross") else None


def quat_conjugate(q: Any) -> Any:
    out = q.clone()
    out[..., 1:4] = -out[..., 1:4]
    return out


def quat_multiply(a: Any, b: Any) -> Any:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    import torch

    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def world_pose_to_frame(
    position_w: Any,
    quaternion_w: Any,
    frame_pos_w: Any,
    frame_quat_w: Any,
) -> tuple[Any, Any]:
    """Express a world pose in the frame given by ``frame_*_w``.

    Kept as its own function because getting this backwards produces a
    plausible pose in the wrong place -- the exact failure class that put
    ER-2's correctly-identified plate 500 m away earlier in this project.
    """
    inv = quat_conjugate(frame_quat_w.unsqueeze(0)).squeeze(0)
    delta = (position_w - frame_pos_w).unsqueeze(0)
    local_pos = quat_rotate(inv.unsqueeze(0), delta).squeeze(0)
    local_quat = quat_multiply(
        inv.unsqueeze(0), quaternion_w.unsqueeze(0)
    ).squeeze(0)
    return local_pos, local_quat


class CuroboGraspPlanner:
    """Thin wrapper: an ER-2 world grasp pose in, joint trajectories out.

    Built lazily and never fatally: every failure path returns ``None`` so
    `reach()` falls back to the servo it uses today. That is the same
    never-abort contract `_live_er_grasp_pose` and `_perception_grasp_target`
    already follow -- a planner outage should cost grasp quality, not the
    episode.
    """

    def __init__(self, robot_yml: str, side: str = "left") -> None:
        import sys

        if CUROBO_SPIKE_PATH not in sys.path:
            sys.path.insert(0, CUROBO_SPIKE_PATH)
        import torch
        import warp as wp  # noqa: F401 -- must precede cuRobo (GOTCHAS)

        from curobo.inverse_kinematics import InverseKinematicsCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.trajectory_optimizer import TrajectoryOptimizerCfg

        self._torch = torch
        self.side = side
        ik_cfg = InverseKinematicsCfg.create(
            robot=robot_yml,
            num_seeds=32,
            self_collision_check=False,
            load_collision_spheres=False,
            position_tolerance=0.005,
        )
        trajopt_cfg = TrajectoryOptimizerCfg.create(
            robot=robot_yml,
            self_collision_check=False,
            load_collision_spheres=False,
        )
        self._planner = MotionPlanner(
            MotionPlannerCfg(
                ik_solver_config=ik_cfg,
                trajopt_solver_config=trajopt_cfg,
            )
        )
        self.tool_frames = list(self._planner.tool_frames)
        self.joint_names = list(self._planner.joint_names)

    def plan(
        self,
        grasp_position_frame: Any,
        grasp_quaternion_frame: Any,
        current_joint_positions: Any,
        *,
        half_open_m: float = DEFAULT_HALF_OPEN_M,
        approach_offset_m: float = -0.10,
    ) -> Any | None:
        """Plan approach/grasp/lift for a grasp pose already in the arm's
        base frame. Returns the cuRobo ``GraspPlanResult`` or ``None``.
        """
        torch = self._torch
        from curobo.types import GoalToolPose, JointState

        n_links = len(self.tool_frames)
        device = grasp_position_frame.device

        # Straddle along the gripper's REAL opening axis, measured from the
        # robot's own kinematics, not assumed.
        #
        # Assuming the tool frame's y axis is what the jaws open along cost a
        # run: FK at the default state puts the two pads at
        # (0.1811, 0.0002, 0.5153) and (0.2136, -0.0002, 0.5253) -- they
        # differ in x, not y. Offsetting along y therefore asks for a pad
        # configuration this gripper cannot make, and plan_grasp answers
        # "Goalset planning returned None", the same status it gives for both
        # pads at one point.
        axis = self._opening_axis(current_joint_positions, device)
        offsets_local = torch.stack(
            (-half_open_m * axis, half_open_m * axis)
        )[:n_links]
        quat_b = grasp_quaternion_frame.unsqueeze(0).expand(n_links, 4)
        offsets_frame = quat_rotate(quat_b, offsets_local)
        positions = (grasp_position_frame.unsqueeze(0) + offsets_frame).view(
            1, 1, n_links, 1, 3
        )
        quats = quat_b.view(1, 1, n_links, 1, 4)

        # Every field must carry the SAME leading batch dim. Passing only
        # `position=(1, dof)` lets JointState default the others to (dof,)
        # and plan_grasp then raises "current_velocity rows (7) !=
        # current_position rows (1)".
        pos = current_joint_positions.view(1, -1).to(device)
        zeros = torch.zeros_like(pos)
        state = JointState(
            position=pos,
            velocity=zeros.clone(),
            acceleration=zeros.clone(),
            jerk=zeros.clone(),
        )
        try:
            return self._planner.plan_grasp(
                GoalToolPose(
                    tool_frames=self.tool_frames,
                    position=positions.contiguous(),
                    quaternion=quats.contiguous(),
                ),
                state,
                grasp_approach_offset=approach_offset_m,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"CUROBO_PLAN_GRASP_RAISED {exc!r}", flush=True)
            return None

    def _opening_axis(self, joint_positions: Any, device: Any) -> Any:
        """Unit vector, in a pad's own frame, along which the jaws separate.

        Derived by forward-kinematics-ing the CURRENT joint state and reading
        the vector between the two pads, expressed in the first pad's frame.
        No convention is assumed and nothing is hardcoded: if the asset's
        tool frames change, this follows them.
        """
        torch = self._torch
        from curobo.types import JointState

        pos = joint_positions.view(1, -1).to(device)
        zeros = torch.zeros_like(pos)
        fk = self._planner.compute_kinematics(
            JointState(
                position=pos,
                velocity=zeros.clone(),
                acceleration=zeros.clone(),
                jerk=zeros.clone(),
            )
        )
        tp = fk.tool_poses
        pads = tp.position.view(-1, 3)
        quats = tp.quaternion.view(-1, 4)
        separation = pads[1] - pads[0]
        norm = torch.linalg.norm(separation)
        if float(norm) < 1e-6:
            # Degenerate: fall back to the tool frame's y, which is at least
            # a consistent choice rather than a NaN.
            return torch.tensor(
                [0.0, 1.0, 0.0], device=device, dtype=pads.dtype
            )
        separation = separation / norm
        # Into the first pad's own frame, so it can be re-applied to whatever
        # orientation the grasp asks for.
        local = quat_rotate(
            quat_conjugate(quats[0].unsqueeze(0)), separation.unsqueeze(0)
        )[0]
        return local / torch.linalg.norm(local)

    @staticmethod
    def waypoints(result: Any, leg: str) -> Any | None:
        """Joint positions for one leg (``approach``/``grasp``/``lift``)."""
        traj = getattr(result, f"{leg}_interpolated_trajectory", None)
        if traj is None or not hasattr(traj, "position"):
            return None
        pos = traj.position
        # [B, G, T, dof] in the probe's output; collapse the leading dims.
        while pos.ndim > 2:
            pos = pos[0]
        return pos

    @staticmethod
    def leg_succeeded(result: Any, leg: str) -> bool:
        flag = getattr(result, f"{leg}_success", None)
        if flag is None:
            return False
        try:
            return bool(flag.flatten()[0])
        except Exception:  # noqa: BLE001
            return bool(flag)
