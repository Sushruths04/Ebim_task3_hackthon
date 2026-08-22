# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Task 3 FSM skills: waypoint navigation logic and the Isaac base adapter.

NavigateTo is pure logic over Pose2D (CPU-tested in tests/test_skills.py).
TmrBaseAdapter is the thin Isaac-side shim that turns the skill's body-frame
twists into TMR steering/wheel joint targets via
scripts/common/tmr_base_control.py -- it is exercised on GPU by
scripts/task3/verify_navigate.py, never by CPU tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from task3_autonomy.navigation import (
    Pose2D,
    base_twist_toward,
    insert_island_detours,
    pose_reached,
    route_via_door,
    wrap_to_pi,
)

# Intermediate waypoints only shape the route around the wall partition, so
# they can be passed loosely; only the final stop uses the strict tolerance.
# 0.10 (down from 0.15): the kitchen lane and the island-east descent both
# have ~0.16 m lateral margin, so a 0.15 m corner cut could clip geometry.
WAYPOINT_PASS_TOLERANCE_M = 0.10

# Arm transit pose for navigation: the default ready pose spans 1.88 m
# across the outboard-mounted arms while both partition crossings measure
# ~1.2 m (probe_arm_tuck.py, sim-dev-g4b 2026-07-17). Probe v3/v4 winner
# "pnn_j6_15_j4_30": settled body-frame width +-0.37 m (vs +-0.75
# default) and forward overhang 0.778 m at link origins (vs 0.885 for the
# v3 fold, which scraped the kitchen island live in nav9); max
# joint-target error 0.009 rad.
TRANSIT_ARM_POSE: dict[str, float] = {
    "left_fr3v2_joint1": 1.57,
    "left_fr3v2_joint2": -0.87,
    "left_fr3v2_joint3": -1.57,
    "left_fr3v2_joint4": -3.0,
    "left_fr3v2_joint5": 0.0,
    "left_fr3v2_joint6": 1.5,
    "left_fr3v2_joint7": 0.785,
    "right_fr3v2_joint1": -1.57,
    "right_fr3v2_joint2": -0.87,
    "right_fr3v2_joint3": 1.57,
    "right_fr3v2_joint4": -3.0,
    "right_fr3v2_joint5": 0.0,
    "right_fr3v2_joint6": 1.5,
    "right_fr3v2_joint7": 0.785,
}


def ramp_arm_pose(
    robot,
    pose: dict[str, float],
    *,
    step,
    ramp_steps: int = 300,
    settle_steps: int = 200,
    on_tick: Callable[[int], None] | None = None,
) -> None:
    """Ramp arm joint position targets to `pose`, calling `step()` per tick.

    GPU-only helper (lazy torch import). `step` must advance the sim one
    tick (write targets, step physics, update scene); targets persist in
    the articulation's buffer afterward, so the arms hold the pose while
    later code drives the base.

    `on_tick`, if given, is called as `on_tick(tick)` right after each
    ramp-phase `step()` (not during settle) -- the caller can use it to
    sample Cartesian EE state per tick, since this function only knows
    about joint targets and has no notion of end-effector geometry
    (SYNC 21/N1: the interpolation below is a raw joint-space LERP with
    no Cartesian guard, so an elbow/wrist can sweep through nearby
    geometry even when both endpoints are individually safe; on_tick is
    how a caller proves or disproves that for a specific tuck).
    """
    if not pose:
        return
    import torch

    names = sorted(pose)
    joint_ids = [robot.joint_names.index(n) for n in names]
    device = robot.data.joint_pos.device
    start = robot.data.joint_pos[0, joint_ids].clone()
    target = torch.tensor(
        [pose[n] for n in names], device=device, dtype=start.dtype
    )
    for tick in range(ramp_steps):
        alpha = (tick + 1) / ramp_steps
        robot.set_joint_position_target(
            ((1.0 - alpha) * start + alpha * target).unsqueeze(0),
            joint_ids=joint_ids,
        )
        step()
        if on_tick is not None:
            on_tick(tick)
    for _ in range(settle_steps):
        robot.set_joint_position_target(
            target.unsqueeze(0), joint_ids=joint_ids
        )
        step()


class NavigateTo:
    """Drive an omnidirectional base through door-aware waypoints to a target.

    Call compute(pose) every control step; it returns (vx, vy, done) in the
    body frame. Yaw is not commanded here -- the caller holds heading with
    tmr_base_control.compensate_yaw_rate().
    """

    def __init__(
        self,
        target_xy: tuple[float, float],
        target_yaw: float | None = None,
        *,
        max_linear_mps: float = 0.5,
        position_kp: float = 1.5,
        position_tolerance_m: float = 0.03,
        yaw_tolerance_rad: float = math.radians(3.0),
        min_creep_mps: float = 0.0,
    ) -> None:
        self.target_xy = target_xy
        self.target_yaw = target_yaw
        self.max_linear_mps = max_linear_mps
        self.position_kp = position_kp
        self.position_tolerance_m = position_tolerance_m
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.min_creep_mps = min_creep_mps
        self._waypoints: list[tuple[float, float]] | None = None
        self._waypoint_index = 0
        self._done = False

    def compute(self, pose: Pose2D) -> tuple[float, float, bool]:
        if self._done:
            return 0.0, 0.0, True
        if self._waypoints is None:
            # T4 (plans/LOOP_PROMPT_VM_A_REV5.md): additive on top of the
            # door-crossing route -- inserts a detour around
            # KITCHEN_ISLAND_BBOX on any leg that would otherwise cut
            # through it (a same-side kitchen move is the common case
            # this matters for; route_via_door's own island-clear logic
            # only covers the door-approach leg, not general kitchen-
            # local travel). A route with no island-crossing leg comes
            # back unchanged.
            self._waypoints = insert_island_detours(
                route_via_door((pose.x, pose.y), self.target_xy)
            )
            self._waypoint_index = 1 if len(self._waypoints) > 1 else 0

        while self._waypoint_index < len(self._waypoints) - 1:
            waypoint = self._waypoints[self._waypoint_index]
            if pose_reached(
                pose,
                waypoint,
                position_tolerance_m=WAYPOINT_PASS_TOLERANCE_M,
            ):
                self._waypoint_index += 1
            else:
                break

        final_leg = self._waypoint_index >= len(self._waypoints) - 1
        if final_leg and pose_reached(
            pose,
            self.target_xy,
            self.target_yaw,
            position_tolerance_m=self.position_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
        ):
            self._done = True
            return 0.0, 0.0, True

        goal = (
            self.target_xy
            if final_leg
            else self._waypoints[self._waypoint_index]
        )
        vx, vy = base_twist_toward(
            pose,
            goal,
            max_linear_mps=self.max_linear_mps,
            position_kp=self.position_kp,
            min_creep_mps=self.min_creep_mps,
        )
        return vx, vy, False


class RotateTo:
    """Rotate the base in place to an absolute world yaw.

    compute(pose) -> (wz_cmd, done). Caller must pass wz_cmd through
    TmrBaseAdapter.apply_twist(0, 0, wz_cmd). Rotation sweeps the robot's
    full footprint (tucked nose 0.78 m / ready arms 0.95 m), so only
    rotate at spots with that much radial clearance.
    """

    def __init__(
        self,
        target_yaw: float,
        *,
        max_yaw_rate: float = 0.5,
        yaw_kp: float = 1.5,
        yaw_tolerance_rad: float = math.radians(2.0),
        min_creep_radps: float = 0.0,
    ) -> None:
        self.target_yaw = target_yaw
        self.max_yaw_rate = max_yaw_rate
        self.yaw_kp = yaw_kp
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.min_creep_radps = min_creep_radps
        self._done = False

    def compute(self, pose: Pose2D) -> tuple[float, bool]:
        """`min_creep_radps` is the rotational twin of
        `base_twist_toward`'s `min_creep_mps`, and it exists for the same
        measured reason.

        `yaw_kp * error` decays to zero as the error shrinks, but the wheel
        drives have a ~2 s velocity-tracking lag (DRIVE_DAMPING=500) -- a
        commanded rate that keeps shrinking faster than the wheels can track
        it never lets real rotation start. Measured on
        `outputs/keep_live_er_run3.log`: the base sat 4.13 degrees from a
        4.0-degree tolerance, `yaw_kp * error` commanded 0.108 rad/s, and the
        base moved 0.0094 rad total over 1000 ticks before the watchdog
        declared it stalled -- a failed `_rotate_to` purely because the
        commanded rate was below what the drives will start from rest.

        Zero by default, so every existing caller is byte-identical; a caller
        that passes a nonzero value gets a sustained, non-decaying target the
        wheels can actually track. Overshoot is bounded by one tick's
        rotation, because the caller stops the instant `done` goes true.
        """
        if self._done:
            return 0.0, True
        error = wrap_to_pi(self.target_yaw - pose.yaw)
        if abs(error) <= self.yaw_tolerance_rad:
            self._done = True
            return 0.0, True
        rate = abs(self.yaw_kp * error)
        if self.min_creep_radps > 0.0:
            rate = max(rate, self.min_creep_radps)
        rate = min(rate, self.max_yaw_rate)
        return math.copysign(rate, error), False


class TmrBaseAdapter:
    """Isaac-side shim: body twist -> TMR steering/wheel joint targets.

    GPU-only (imports torch-backed helpers lazily); verified by
    scripts/task3/verify_navigate.py, not by CPU tests.
    """

    # Wheel velocity-drive damping the wheels actually need: 5.0 left them
    # at ~7% of target; 500 tracks within 2 s (probe_base_drive.py,
    # sim-dev-g4b 2026-07-17). Written directly to PhysX here because the
    # same value in robot_actuator_cfg_specs did NOT reach the sim (the two
    # live runs crawled identically before and after that config change) --
    # the runtime tensor write is the delivery mechanism proven by the
    # probe. The readback print below records what the sim had, as
    # evidence for root-causing the config path later.
    DRIVE_DAMPING = 500.0

    def __init__(self, robot, *, num_envs: int, device: str) -> None:
        import tmr_base_control as base
        import torch

        self._base = base
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        self.steering_ids, self.drive_ids = base.find_drive_joint_ids(
            robot.joint_names
        )
        self._hold_yaw = base.get_root_yaw(robot)

        sim_damping = getattr(robot.data, "joint_damping", None)
        if sim_damping is not None:
            damping_values = [
                round(float(sim_damping[0, i]), 3) for i in self.drive_ids
            ]
            print(
                "TmrBaseAdapter: sim wheel damping before override: "
                f"{damping_values}",
                flush=True,
            )
        robot.write_joint_damping_to_sim(
            torch.full(
                (num_envs, len(self.drive_ids)),
                self.DRIVE_DAMPING,
                device=device,
            ),
            joint_ids=self.drive_ids,
        )
        print(
            f"TmrBaseAdapter: wheel drive damping set to {self.DRIVE_DAMPING}",
            flush=True,
        )

    def pose(self) -> Pose2D:
        position = self.robot.data.root_pos_w[0]
        return Pose2D(
            float(position[0]),
            float(position[1]),
            self._base.get_root_yaw(self.robot),
        )

    def apply_twist(
        self,
        vx: float,
        vy: float,
        wz_cmd: float = 0.0,
        *,
        hold_heading: bool = False,
        telemetry: list[dict] | None = None,
    ) -> None:
        """Body twist -> wheel targets. Nonzero wz_cmd rotates in place
        (heading hold re-anchors to wherever the rotation ends).

        `telemetry` (REV16 Phase D prep, plans/SYNC.md 2026-08-07 REV14
        T4 follow-up "rotate_west code-read hypothesis"): when given, one
        dict is appended per call with the PRE-tick steering angle, the
        computed steering target, and the resulting steering error --
        the exact per-tick data that SYNC entry's "what would confirm or
        refute this" step asked for (a real steering-realignment cost vs
        a steering-joint gain/damping issue, analogous to the already-
        fixed DRIVE_DAMPING). `None` (default): zero behavior change,
        zero overhead -- unverified on GPU as of this commit.
        """
        wz, self._hold_yaw = self._base.compensate_yaw_rate(
            self.robot,
            vx,
            vy,
            wz_cmd,
            self._hold_yaw,
            manual_rotation=abs(wz_cmd) > 1.0e-4,
            hold_while_stopped=hold_heading,
        )
        current_angle_pre = None
        if telemetry is not None:
            current_angle_pre = [
                float(v)
                for v in self.robot.data.joint_pos[0, self.steering_ids]
            ]
        steering_targets, drive_targets = self._base.compute_drive_targets(
            self.robot,
            self.steering_ids,
            vx,
            vy,
            wz,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.robot.set_joint_position_target(
            steering_targets, joint_ids=self.steering_ids
        )
        self.robot.set_joint_velocity_target(
            drive_targets, joint_ids=self.drive_ids
        )
        if telemetry is not None:
            targets = [float(v) for v in steering_targets[0]]
            telemetry.append(
                {
                    "wz_cmd": wz_cmd,
                    "wz_after_compensate": wz,
                    "current_angle_pre": current_angle_pre,
                    "steering_targets": targets,
                    "steering_error": [
                        wrap_to_pi(t - c)
                        for t, c in zip(targets, current_angle_pre)
                    ],
                    # REV16 Phase D follow-up: run1's telemetry (steering
                    # only) refuted the swerve-drive steering-realignment
                    # hypothesis outright -- steering_error converges to
                    # ~0 within ~1s and wz_after_compensate stays at the
                    # full commanded -0.5 rad/s for the remaining ~14s,
                    # yet the base only rotated 0.133 rad total. The
                    # bottleneck must be downstream of wz: either the
                    # commanded wheel velocities (drive_targets) or
                    # whether the wheels actually achieve them
                    # (joint_vel, the same DRIVE_DAMPING mechanism this
                    # class already fixed once for translation -- unknown
                    # whether it holds for the much smaller wheel speeds
                    # a pure rotation commands).
                    "drive_targets": [
                        float(v) for v in drive_targets[0]
                    ],
                    "drive_joint_vel_actual": [
                        float(v)
                        for v in self.robot.data.joint_vel[0, self.drive_ids]
                    ],
                    "root_ang_vel_z": float(
                        self._base.get_root_yaw_rate(self.robot)
                    ),
                }
            )
