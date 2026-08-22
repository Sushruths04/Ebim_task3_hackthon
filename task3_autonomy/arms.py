# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dual-arm manipulation skills for the Task 3 autonomous FSM.

The pure helpers in this module build the same incremental
``TeleopCommand`` consumed by keyboard teleoperation. ``DualArmController``
then feeds the resulting ``CartesianTargetTracker`` targets through the
proven Lula IK and joint-target composition stack. Isaac imports remain lazy
so command math and hold predicates stay CPU-testable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from task3_autonomy.rotations import rpy_from_quaternion

# ChangingTek AG2F primary revolute-joint positions. This linkage follows the
# same convention as the original FR3 finger controller: zero is closed and
# increasing the position opens the fingers (USD limit 0..1 rad).
GRIPPER_CLOSED_RAD = 0.0
# 2026-08-20 (real bug, GPU-measured, not guessed): this was 0.9, past
# the driven joint's own USD-authored upper limit -- measured directly
# via `joint_pos_limits` (scripts/task3/probe_gripper_joint_limits.py):
# [0.0, 0.8203047513961792] on this asset, both sides. Every caller of
# `release()`/`open_before_approach` that gates on
# `abs(position - GRIPPER_OPEN_RAD) <= tolerance_rad` was therefore
# guaranteed to fail 100% of the time regardless of stiffness/timeout
# tuning -- confirmed: 28/28 real failures across three full GPU runs,
# gripper_position_rad stiffness-invariant at 0.8203 every time. Also
# affected `honest_hold`'s `gripper_rad < GRIPPER_OPEN_RAD` check
# (task3_pipeline/world_isaac.py), which was vacuously true against an
# unreachable ceiling -- the joint's own authored limit is the only
# value that makes that check mean anything.
GRIPPER_OPEN_RAD = 0.82

DEFAULT_POSITION_TOLERANCE_M = 0.02
DEFAULT_ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
DEFAULT_HOLD_MIN_POSITION_RAD = 0.05
# The jaws' OPEN limit, used as the hold band's reference point. Callers with
# a live robot should pass the authored value from
# `DualArmController._gripper_position_upper_limit()` instead of relying on
# this; it exists so the pure helpers stay callable without one.
#
# 2026-08-21: this was 1.05 -- a hand-picked number sitting PAST the joint's
# own mechanical limit, so a gripper jammed fully open (measured 1.0145 under
# contact-induced overshoot) scored as a successful hold.
DEFAULT_GRIPPER_OPEN_RAD = 0.82

# How far the measured position must lag the commanded target before a tick
# counts as CONTACT -- something is resisting closure. Defined here, above
# its first use, because `gripper_holds_object` derives the hold band's
# upper margin from it: jaws within one contact-deflection of fully open
# cannot be meaningfully loaded.
DEFAULT_CONTACT_ERROR_RAD = 0.03


def grasp_lift_gate_passed(
    *,
    holding: bool,
    held_ticks: int,
    needed_ticks: int,
    lifted_m: float,
    min_lift_m: float,
) -> bool:
    """Require measured object retention, height, and continuous hold.

    End-effector convergence is diagnostic only: a valid physical lift can
    meet the object-space goal even when the wrist stops short of a more
    ambitious Cartesian target.
    """
    return holding and held_ticks >= needed_ticks and lifted_m >= min_lift_m


def linear_ramp_target(
    start: float, end: float, completed_steps: int, total_steps: int
) -> float:
    """Return a clamped linear ramp target for deterministic soft closure."""
    if not all(math.isfinite(value) for value in (start, end)):
        raise ValueError("ramp endpoints must be finite")
    if total_steps <= 0 or completed_steps < 0:
        raise ValueError("ramp steps must be positive")
    alpha = min(1.0, completed_steps / total_steps)
    return start + (end - start) * alpha


def synchronized_drag_targets(
    arm_start_y: float,
    anchor_start_y: float,
    distance: float,
    completed_steps: int,
    total_steps: int,
) -> tuple[float, float]:
    """Advance an arm push target and a base hold anchor by one shared offset.

    Both must move north together so the arm's commanded reach relative to
    the base never grows past the proven envelope: the Step 1 trial 1 root
    cause was a single unsynchronized reach ~1.0 m from stance, well past the
    proven ~0.83 m dead-ahead ceiling. Sharing one ramp offset guarantees the
    arm/base separation stays exactly constant for every step.
    """
    offset = linear_ramp_target(0.0, distance, completed_steps, total_steps)
    return arm_start_y + offset, anchor_start_y + offset


def ordered_joint_targets(
    targets: Mapping[str, float], joint_names: Sequence[str]
) -> list[float] | None:
    """Convert Lula's immutable name mapping into composer joint order."""
    if not targets:
        return None
    try:
        values = [float(targets[name]) for name in joint_names]
    except KeyError as error:
        raise ValueError(
            f"IK result is missing joint {error.args[0]}"
        ) from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("IK joint targets must be finite")
    return values


def one_step_reach_command(
    current_base_target: Any,
    world_target: Any,
    base_position: Sequence[float],
    base_orientation_wxyz: Sequence[float],
    *,
    side: str,
    timestamp: float = 0.0,
) -> Any:
    """Build the single incremental command that lands on ``world_target``.

    ``CartesianTargetTracker`` stores a base-frame target and left-multiplies
    its orientation by the command delta. Therefore the exact delta is
    ``target * inverse(current)``. Frame conversion and quaternion operations
    deliberately reuse ``teleop_targets`` rather than introducing a second
    implementation.
    """
    from teleop_commands import PoseDelta, TeleopCommand
    from teleop_targets import (
        _normalize_quaternion,
        _quaternion_conjugate,
        _quaternion_multiply,
        pose_world_to_base,
    )

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    target_base = pose_world_to_base(
        world_target, base_position, base_orientation_wxyz
    )
    current_orientation = _normalize_quaternion(
        current_base_target.orientation_wxyz
    )
    delta_quaternion = _quaternion_multiply(
        target_base.orientation_wxyz,
        _quaternion_conjugate(current_orientation),
    )
    delta = PoseDelta(
        translation=tuple(
            target - current
            for target, current in zip(
                target_base.position, current_base_target.position
            )
        ),
        rotation_rpy=rpy_from_quaternion(delta_quaternion),
    )
    kwargs = {"left_pose": delta} if side == "left" else {"right_pose": delta}
    return TeleopCommand(
        timestamp=timestamp,
        source="task3_autonomy.reach",
        active=True,
        **kwargs,
    )


def gripper_holds_object(
    position_rad: float,
    *,
    min_position_rad: float = DEFAULT_HOLD_MIN_POSITION_RAD,
    max_position_rad: float = DEFAULT_GRIPPER_OPEN_RAD,
    hold_margin_rad: float = DEFAULT_CONTACT_ERROR_RAD,
) -> bool:
    """Whether the jaws stopped somewhere an object could be holding them.

    ``max_position_rad`` is the jaws' **OPEN** limit -- the authored upper
    joint limit, which is what every caller already passes (via
    ``DualArmController._gripper_position_upper_limit``). The band's real
    upper bound is that value minus ``hold_margin_rad``.

    That margin is the whole point of this function and it was missing.
    Callers were correctly derived off the measured open limit (0.8203 rad
    on this asset) and then compared against it directly, so the band that
    means "something is wedged between the fingers" ran to within
    **0.0003 rad** of "the fingers are wide open, holding nothing". A
    gripper reading 0.819 -- open, closed on air -- satisfied
    ``0.05 < 0.819 < 0.8203`` and was scored as a successful grasp. The
    accompanying ``gripper_rad < GRIPPER_OPEN_RAD`` guard in
    ``world_isaac.honest_hold`` did not help: 0.819 < 0.82 too.

    ``hold_margin_rad`` is derived, not chosen: it is
    ``DEFAULT_CONTACT_ERROR_RAD``, this codebase's own measured definition
    of how far the measured position must lag the commanded target before a
    tick counts as contact. Jaws within one contact-deflection of fully
    open cannot be meaningfully loaded. It is also confirmed permissive
    enough for a real grip: the 2026-08-20 gamepad session's genuine
    cup-wall hold measured **0.783 rad**, comfortably inside the resulting
    ``0.05 < pos < 0.7903`` band.
    """
    values = (
        position_rad,
        min_position_rad,
        max_position_rad,
        hold_margin_rad,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gripper positions must be finite")
    if hold_margin_rad < 0.0:
        raise ValueError("gripper hold margin must be non-negative")
    upper = max_position_rad - hold_margin_rad
    if min_position_rad < 0.0 or upper <= min_position_rad:
        raise ValueError("gripper hold bounds must be positive and ordered")
    return min_position_rad < position_rad < upper



# REV13 T4-followup (plans/SYNC.md 2026-08-07): T4's naive
# `hold_target_on_contact` (freeze on ANY tick crossing `contact_error_rad`)
# failed on real GPU -- it froze at `held_target_rad=0.92332`, essentially
# still fully open (`GRIPPER_OPEN_RAD=0.9`), so nothing was actually
# gripped. Comparing that failure's raw tick data against T3's real,
# eventually-lost-but-genuine contact (`proofs/2026-08-06_t3_bowl2_close_
# telemetry` vs `..._t4_bowl2_close_fix_reverted`) shows the discriminating
# signal T4 missed: T3's contact was detected at tick 2 with the commanded
# target already at 0.647 rad (solidly into the closing ramp, and almost
# exactly this project's own upper proven-hold value, 0.6472); T4's
# "contact" triggered at tick 4 with the target still at 0.923 rad --
# barely past the gripper's own open rest position, before the ramp had
# meaningfully attempted to close at all. Freezing there cannot produce a
# useful grip regardless of how real the detected resistance is. This
# constant gates freezing on how far the ramp had already progressed when
# contact was seen, not just whether error crossed a threshold.
#
# First value tried (0.8, with margin on both the 0.647/0.923 data points)
# was tested live on real GPU (2 fresh episodes, bowl2/left, the proven
# stance) and found too loose: one run froze on a shallow tick-1 contact
# at target=0.74052 -- still barely closed (final `gripper_rad: 0.7747`,
# `lift.ik_ok: False`), the same failure SHAPE as T4's original bug at a
# slightly lower number, not a real fix. Tightened to 0.65 -- just above
# T3's own real value (0.647) with almost no slack, deliberately, since
# the evidence so far is that ANY meaningful slack above T3's real number
# admits shallow false positives. The other of the 2 fresh episodes came
# close under the 0.8 setting (`gripper_rad: 0.2137` vs the 0.25 gate,
# `object_rise_m: -0.0001`) without even needing the loose cutoff, which
# is consistent with 0.65 being tight enough to still pass real contact
# through.
DEFAULT_CONTACT_FREEZE_MAX_TARGET_RAD = 0.65


def run_gripper_close_ramp(
    *,
    start_position: float,
    end_position: float,
    total_ticks: int,
    ramp_ticks: int,
    set_target: Callable[[float], None],
    measured_position: Callable[[], float],
    advance: Callable[[], None],
    contact_error_rad: float = DEFAULT_CONTACT_ERROR_RAD,
    hold_target_on_contact: bool = False,
    freeze_on_stall: bool = True,
    stall_ticks_required: int = 8,
    stall_ignore_ticks: int = 20,
    stall_min_open_rad: float = 0.02,
    stall_lookback_ticks: int = 1,
    contact_freeze_max_target_rad: float = (
        DEFAULT_CONTACT_FREEZE_MAX_TARGET_RAD
    ),
    hold_max_position_rad: float = DEFAULT_GRIPPER_OPEN_RAD,
    telemetry: dict[str, Any] | None = None,
) -> bool:
    """Drive a gripper closed tick-by-tick, recording per-tick telemetry.

    REV13 T2 (plans/SYNC.md 2026-08-06): a close that never starts and a
    close that starts and slips were indistinguishable from the outside
    -- both this project's proven successes (gripper_rad 0.2979, 0.6472)
    and its 9 T7 failures (<=0.1292) went through the exact same call
    with no before/after way to see WHY the final residual differed
    (REV13 T1's diff table). This is the pure tick loop `DualArmController
    .grasp()` delegates to, so it is CPU-testable with plain callables --
    no real robot/tracker required, matching this module's existing
    pure-helper convention (`linear_ramp_target`, `gripper_holds_object`).

    ``contact_error_rad``: a tick counts as "contact" when the measured
    position lags the commanded ramp target by more than this -- i.e.
    something is resisting closure. Three tick-level outcomes fall out
    of that alone: the fingers never meet resistance and land closed at
    the ramp's end (``closed_no_contact``); resistance appears and never
    releases before the ramp ends (``contact_sustained`` -- the proven-
    band shape, e.g. 0.2979/0.6472); resistance appears then the position
    converges back down toward the ramp target before the end
    (``contact_lost`` -- caught the object, then it slipped or was pushed
    clear).

    ``hold_target_on_contact`` (REV13 T4, plans/SYNC.md 2026-08-06): T3's
    tick-level trace of a real GPU miss showed `contact_lost` is not a
    brief near-touch -- the fingers were shoved past their own open rest
    limit at tick 2, held under real resistance for ~190/300 ticks, and
    only collapsed to fully-closed-on-air in the final ~50 ticks, exactly
    while the commanded ramp kept forcing the target toward 0 rad long
    after contact was already established. The current default (always
    ramp all the way to `end_position` regardless of contact) actively
    fights and eventually destroys a grip it has already made. When
    True, the commanded target FREEZES at whatever it was the tick
    contact was first detected, instead of continuing toward
    `end_position` -- stop squeezing once something is already caught,
    the way a compliant/force-limited real gripper would.

    ``contact_freeze_max_target_rad`` (REV13 T4-followup, plans/SYNC.md
    2026-08-07): T4's plain `hold_target_on_contact` shipped and FAILED on
    real GPU -- it froze at a commanded target of 0.923 rad, essentially
    still fully open, because contact was flagged the instant error
    crossed `contact_error_rad` regardless of how little the ramp had
    progressed. A single-tick error threshold cannot tell "the fingers
    are genuinely caught on the object, X rad into a real close" apart
    from "the fingers barely started moving and something is already in
    the way at the open end" -- both look identical to that check alone.
    This second gate uses the one signal that DOES separate this
    project's real proven-band freeze point (T3: target=0.647 rad,
    almost exactly the upper proven hold of 0.6472) from its real failed
    freeze point (T4: target=0.923 rad, barely past `GRIPPER_OPEN_RAD`):
    how far into the closing ramp contact was detected. Only freeze when
    the commanded target at the contact tick is already at or below this
    value; a contact flagged earlier than that is still recorded (so
    `contact_tick`/telemetry are unaffected) but does not freeze -- the
    ramp keeps commanding toward `end_position` exactly as the unfixed
    default already does, which is the behavior both of this project's
    two actually-proven holds went through.

    ``stall_lookback_ticks`` (2026-08-16, cup): raising
    `contact_freeze_max_target_rad` to cover a shallow contact (a cup rim
    stalled at measured=0.89, deep_enough now true) still froze nothing --
    `outputs/task3_verify_grasp_lift/close_trace/close_ramp_ticks.json`
    shows why. The position is not a clean stall, it JITTERS a few
    thousandths of a radian tick to tick under sustained contact (e.g.
    ticks 52-58: 0.8905, 0.89042, 0.89742, 0.90636, 0.89852, 0.89159,
    0.90286 -- net flat, but individual ticks go up and down), while
    `error_rad` grows steadily the whole time (-0.16 at tick 52 to -0.46 by
    tick 90), proving the object IS caught throughout. Comparing against
    only the immediately-previous tick (the default, `stall_lookback_ticks
    =1`) reads that jitter as "closing" on every uptick, resetting
    `stalled_ticks` before it ever reaches `stall_ticks_required` -- the
    ceiling was never the bottleneck, the single-tick comparison was.
    Comparing against the measurement from `stall_lookback_ticks` ago
    instead filters the jitter: net progress over the window still reads
    as closing, net flat-under-resistance reads as stalled. Default 1
    reproduces the exact prior behavior for every existing caller;
    NOT YET n>=3 verified for any value > 1.
    """
    if total_ticks <= 0 or ramp_ticks <= 0:
        raise ValueError("total_ticks and ramp_ticks must be positive")
    if stall_lookback_ticks < 1:
        raise ValueError("stall_lookback_ticks must be >= 1")
    ticks: list[dict[str, float | int]] = []
    contact_tick: int | None = None
    held_target: float | None = None
    measured_history: list[float] = []
    stalled_ticks = 0
    for completed in range(1, total_ticks + 1):
        if held_target is not None:
            target = held_target
        else:
            target = linear_ramp_target(
                start_position, end_position, completed, ramp_ticks
            )
        set_target(target)
        advance()
        measured = measured_position()
        error = target - measured
        # STALL DETECTION: the jaws have met the object when the measured
        # position stops following a still-closing command. That is a
        # stronger signal than `abs(error) > contact_error_rad`, which fires
        # on servo lag within a handful of ticks regardless of what is
        # there -- `close_contact_tick` was 4-9 in every close of this
        # project, for every object, held or not.
        #
        # Measured, stiff_1: with gripper stiffness raised to 60 the jaws
        # finally close with authority, stop on the spoon at 0.4077 rad --
        # and then keep going to 0.0011, ejecting it before the carry. The
        # ramp needs to STOP where the object stopped it.
        if (
            freeze_on_stall
            and held_target is None
            and len(measured_history) >= stall_lookback_ticks
            and completed > stall_ignore_ticks
        ):
            reference = measured_history[-stall_lookback_ticks]
            closing = measured < reference - 1e-6
            # Same progress gate as hold_target_on_contact, and for the same
            # reason: a stall while the jaws are still near their open rest
            # position is not an object, it is the T4 failure
            # (`proofs/2026-08-06_t4_bowl2_close_fix_reverted` froze at
            # 0.923 rad and gripped nothing). Only a stall the ramp has
            # actually closed into counts.
            deep_enough = measured <= contact_freeze_max_target_rad
            if (
                not closing
                and deep_enough
                and measured > end_position + stall_min_open_rad
            ):
                stalled_ticks += 1
                if stalled_ticks >= stall_ticks_required:
                    held_target = measured
            else:
                stalled_ticks = 0
        measured_history.append(measured)
        if contact_tick is None and abs(error) > contact_error_rad:
            contact_tick = completed
            if (
                hold_target_on_contact
                and target <= contact_freeze_max_target_rad
            ):
                held_target = target
        ticks.append(
            {
                "tick": completed,
                "commanded_target_rad": round(target, 5),
                "measured_position_rad": round(measured, 5),
                "error_rad": round(error, 5),
            }
        )
    final = ticks[-1]
    final_measured = float(final["measured_position_rad"])
    final_error = float(final["error_rad"])
    if contact_tick is None:
        outcome = "closed_no_contact"
    elif abs(final_error) > contact_error_rad:
        outcome = "contact_sustained"
    else:
        outcome = "contact_lost"
    holding = gripper_holds_object(
        final_measured, max_position_rad=hold_max_position_rad
    )
    if telemetry is not None:
        telemetry["ticks"] = ticks
        telemetry["tick_count"] = len(ticks)
        telemetry["contact_tick"] = contact_tick
        telemetry["final_residual_rad"] = round(final_measured, 5)
        telemetry["final_error_rad"] = round(final_error, 5)
        telemetry["outcome"] = outcome
        telemetry["holding"] = holding
        telemetry["hold_target_on_contact"] = hold_target_on_contact
        telemetry["contact_freeze_max_target_rad"] = (
            contact_freeze_max_target_rad
        )
        telemetry["held_target_rad"] = (
            round(held_target, 5) if held_target is not None else None
        )
    return holding


def _quaternion_angle_error(
    measured: Sequence[float], target: Sequence[float]
) -> float:
    from teleop_targets import _normalize_quaternion

    measured_q = _normalize_quaternion(tuple(measured))
    target_q = _normalize_quaternion(tuple(target))
    dot = abs(sum(a * b for a, b in zip(measured_q, target_q)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


class DualArmController:
    """Absolute-world manipulation interface over the teleop/Lula runtime."""

    def __init__(
        self,
        robot: Any,
        simulation_app: Any,
        gripper: str | None = None,
    ) -> None:
        from dual_arm_lula import (
            LEFT_ARM_JOINTS,
            RIGHT_ARM_JOINTS,
            create_raw_dual_arm_lula,
        )
        from scene_robot_room_keyboard import (
            enable_motion_generation_extension,
            measured_position_targets,
            robot_root_world_pose,
        )
        from teleop_targets import (
            CartesianTargetTracker,
            TargetLimits,
            compose_position_targets,
            discover_joint_groups,
            pose_base_to_world,
            position_target_subset,
        )

        import omni.kit.app

        self._CartesianTargetTracker = CartesianTargetTracker
        self._TargetLimits = TargetLimits
        self._compose = compose_position_targets
        self._pose_base_to_world = pose_base_to_world
        self._subset = position_target_subset
        self._measured_position_targets = measured_position_targets
        self._root_pose = robot_root_world_pose
        self._left_arm_joint_names = LEFT_ARM_JOINTS
        self._right_arm_joint_names = RIGHT_ARM_JOINTS

        self.robot = robot
        enable_motion_generation_extension(
            omni.kit.app.get_app().get_extension_manager()
        )
        self.joint_groups = discover_joint_groups(
            robot.joint_names, gripper=gripper
        )
        self._configure_arm_joint_gains(robot)
        self._default_gripper_effort_limits = {
            side: self._gripper_effort_limit(side)
            for side in ("left", "right")
        }
        self._position_targets = measured_position_targets(robot)
        self._ik = create_raw_dual_arm_lula(
            robot.joint_names,
            lambda: robot.data.joint_pos[0].detach().cpu().numpy(),
        )
        self._tracker = None
        self.sync_targets_from_measured()

    # VM_A_BRIEF A2 (GATE A1 CONFIRMED, plans/VM_A_LOG.md): arm joint
    # stiffness/damping were never set anywhere in this code path --
    # reach() commands PD position targets via set_joint_position_target,
    # but nothing tunes the drive that tracks them, unlike the wheels
    # (TmrBaseAdapter.__init__ in skills.py, same pattern mirrored here).
    # A GPU run's joint_tracking_trace showed the wrist joint (index 6)
    # sitting 0.18-0.27 rad off its commanded target for a full 6s
    # 100%-IK-solving push_contact attempt -- consistent with an
    # under-tuned drive, not an IK/reachability failure.
    #
    # task1_isaacsim/assets/embodiments/fr3duo_mobile/isaac_joint_drives
    # .yaml documents this same FR3 duo asset family's own designed arm
    # gains (extracted from the source USD, joint-numbered identically to
    # LEFT_ARM_JOINTS/RIGHT_ARM_JOINTS): joints 1-4 (heavy, max_force
    # 87 N*m) at stiffness 625/damping 60 (zeta~1.2, slightly
    # overdamped); joints 5-7 (light, max_force 12 N*m) at stiffness
    # 625/damping 40 (zeta~0.8, near-critical) -- not a blind guess, the
    # asset's own prior calibration. Applying it as variant 1.
    _ARM_HEAVY_JOINT_COUNT = 4
    _ARM_HEAVY_STIFFNESS = 625.0
    _ARM_HEAVY_DAMPING = 60.0
    _ARM_LIGHT_STIFFNESS = 625.0
    _ARM_LIGHT_DAMPING = 40.0

    # 2026-08-15: DAMPING ONLY, light joints only -- a strictly narrower
    # change than the stiffness+damping variant above, which T2 disabled on
    # evidence. Derived from the joint's own torque ceiling, not fitted.
    #
    # Measured (run 11, a `recenter` that solved IK on 800/800 ticks):
    # `position_error_trace` converges by tick 200 and then plateaus
    # exactly -- (200, 0.0302) (400, 0.0316) (600, 0.0317) (800, 0.0318).
    # 600 further ticks buy 0.0016 m. One joint owns that plateau: joint 7
    # sits 0.4168 rad from its command at tick 800 while every other joint
    # is inside 0.1, moving from actual +0.0262 to -0.0260 against a
    # command of -0.55 -> -0.44. It barely moves.
    #
    # It is not a limit (`left_fr3v2_joint7` is +/-3.05083 rad, command
    # -0.44) and not IK (800/800 solved). It is the drive. At a 0.42 rad
    # error the PD term `stiffness * error` is 2100 N*m against a 12 N*m
    # effort limit, so the joint is torque-SATURATED for the whole motion
    # and its terminal velocity is `effort / damping`, independent of
    # stiffness:
    #
    #     native damping 500 -> 12/500 = 0.024 rad/s -> 0.42 rad in 17.5 s
    #     this   damping  40 -> 12/40  = 0.300 rad/s -> 0.42 rad in  1.4 s
    #
    # `arms.reach` runs 4 s. That is the whole story: the joint cannot
    # arrive, every reach ends 2-3 cm out, and 3 cm of lateral error on a
    # cup rim is the difference between straddling it and closing beside
    # it.
    #
    # Stiffness is deliberately left native (5000): it is not what is
    # saturating, and near the target the small-error torque `5000 * err`
    # is what actually holds position. Damping 40 against stiffness 5000 is
    # still overdamped for these light links (zeta ~1.6 at I~0.03 kg*m^2),
    # so this does not trade steady-state error for oscillation.
    #
    # 40.0 is the asset's own designed value for joints 5-7
    # (task1_isaacsim/assets/embodiments/fr3duo_mobile/isaac_joint_drives
    # .yaml, extracted from the source USD), not a number chosen to make a
    # run pass.
    #
    # NOT YET n>=3 VERIFIED. C9 applies.
    #
    # 2026-08-16: trying the intermediate value 8939f3a's own commit
    # message prescribed as the next attempt (100-200, i.e. 0.06-0.12
    # rad/s), instead of damping 40 (0.30 rad/s, too fast -- tracked the
    # slow phases well but made the gentle ramp diverge, aborting at tick
    # 502/600) or native 500 (0.024 rad/s, too slow -- every reach settles
    # 2-3 cm out AND the live-ER run this session hit the exact same
    # gentle-ramp divergence at tick 588/600 with damping 40 never even
    # applied, ramp_deviation_m 0.0842 vs 0.08 tolerance,
    # outputs/keep_liveer_cup_v2_postfix.log). 150 is the midpoint:
    # 12/150 = 0.08 rad/s, roughly midway between the two measured
    # failure points. Enabled on left+right (the fix was proven with only
    # "right" evidence in the commit, but this session's failure was on
    # "left" -- both arms grasp objects in run_task3, not just the one
    # verify_grasp_lift.py hardcodes).
    #
    # Must clear C9 (n>=3) on BOTH the recenter error and the ramp
    # deviation before being believed -- one clean run is not evidence.
    _ARM_LIGHT_DAMPING_ONLY = 150.0
    _ARM_LIGHT_DAMPING_ONLY_SIDES: tuple[str, ...] = ("left", "right")

    # T2 (plans/VM_B_LOG.md, SYNC 33 -> T2): fully disabled, on evidence,
    # not merely reverted blind. VM_A_BRIEF A2's own recommendation was to
    # scope this to the right arm only (its rationale: Stage 4's
    # push_contact measured wrist drift on the right arm, and it assumed
    # Stage 1's grasp uses the left arm). That assumption is FALSE for this
    # project's own Stage 1 gate script: `scripts/task3/verify_grasp_lift.py`
    # drives the grasp with the RIGHT arm exclusively (every
    # `arms.set_arm_target`, `position_error`, `gripper*`, `servo_arm` call
    # is hardcoded "right"). Measured (exact numbers: plans/SYNC.md SYNC
    # 33 -> T2): scoping the override to "right" reproduced the ORIGINAL
    # ~0.17m pregrasp shortfall -- both stages drive the same arm, so
    # scoping cannot protect one without breaking the other. Scoping to
    # "left" instead (never touching the arm Stage 1 needs untouched)
    # restored pregrasp to ~0.014m error, matching the proven
    # disabled-override baseline. Since scoping to "left" is functionally
    # identical to full disable for every current caller (nothing today
    # drives the left arm with precision this override would help or hurt),
    # disabling outright is the simpler, equally-evidenced choice -- not an
    # arbitrary populated tuple left over from an untested guess.
    # Re-enabling for Stage 4's right-arm push is VM A's call to make with
    # their own gate evidence; flagged to them in SYNC 33 -> T2.
    _ARM_GAIN_OVERRIDE_SIDES: tuple[str, ...] = ()

    def _configure_arm_joint_gains(self, robot: Any) -> None:
        import torch

        sim_stiffness = getattr(robot.data, "joint_stiffness", None)
        sim_damping = getattr(robot.data, "joint_damping", None)
        # R5 T2: unconditional, read-only -- logs the native (sim-default)
        # gains regardless of `_ARM_GAIN_OVERRIDE_SIDES`, which is empty
        # today. No override is applied from this block; it exists purely
        # to answer "are native gains under-damped for a loaded arm" with
        # real numbers instead of guessing.
        if sim_stiffness is not None and sim_damping is not None:
            for side in ("left", "right"):
                arm_ids = list(getattr(self.joint_groups, f"{side}_arm"))
                native = [
                    (
                        round(float(sim_stiffness[0, i]), 3),
                        round(float(sim_damping[0, i]), 3),
                    )
                    for i in arm_ids
                ]
                print(
                    f"GAINSDBG native {side} arm (stiffness, damping): "
                    f"{native}",
                    flush=True,
                )
        # Damping-only correction for the light wrist joints. Separate from
        # the stiffness+damping block below on purpose: that variant was
        # disabled on evidence, this one is a different and much narrower
        # claim -- see `_ARM_LIGHT_DAMPING_ONLY`'s comment for the
        # effort/damping arithmetic it comes from.
        for side in self._ARM_LIGHT_DAMPING_ONLY_SIDES:
            arm_ids = list(getattr(self.joint_groups, f"{side}_arm"))
            light_ids = arm_ids[self._ARM_HEAVY_JOINT_COUNT :]
            if not light_ids:
                continue
            robot.write_joint_damping_to_sim(
                torch.full(
                    (1, len(light_ids)),
                    self._ARM_LIGHT_DAMPING_ONLY,
                    device=robot.data.joint_pos.device,
                ),
                joint_ids=light_ids,
            )
            print(
                f"ARM_LIGHT_DAMPING_ONLY {side} joints5-7 damping -> "
                f"{self._ARM_LIGHT_DAMPING_ONLY} "
                f"(stiffness left native; max joint speed "
                f"effort/damping = 12/{self._ARM_LIGHT_DAMPING_ONLY:.0f} = "
                f"{12.0 / self._ARM_LIGHT_DAMPING_ONLY:.3f} rad/s)",
                flush=True,
            )

        for side in self._ARM_GAIN_OVERRIDE_SIDES:
            arm_ids = list(getattr(self.joint_groups, f"{side}_arm"))
            if sim_stiffness is not None and sim_damping is not None:
                before = [
                    (
                        round(float(sim_stiffness[0, i]), 3),
                        round(float(sim_damping[0, i]), 3),
                    )
                    for i in arm_ids
                ]
                print(
                    f"DualArmController: sim {side} arm "
                    f"(stiffness, damping) before override: {before}",
                    flush=True,
                )
            heavy_ids = arm_ids[: self._ARM_HEAVY_JOINT_COUNT]
            light_ids = arm_ids[self._ARM_HEAVY_JOINT_COUNT :]
            device = robot.data.joint_pos.device
            robot.write_joint_stiffness_to_sim(
                torch.full(
                    (1, len(heavy_ids)),
                    self._ARM_HEAVY_STIFFNESS,
                    device=device,
                ),
                joint_ids=heavy_ids,
            )
            robot.write_joint_damping_to_sim(
                torch.full(
                    (1, len(heavy_ids)), self._ARM_HEAVY_DAMPING, device=device
                ),
                joint_ids=heavy_ids,
            )
            robot.write_joint_stiffness_to_sim(
                torch.full(
                    (1, len(light_ids)),
                    self._ARM_LIGHT_STIFFNESS,
                    device=device,
                ),
                joint_ids=light_ids,
            )
            robot.write_joint_damping_to_sim(
                torch.full(
                    (1, len(light_ids)), self._ARM_LIGHT_DAMPING, device=device
                ),
                joint_ids=light_ids,
            )
            print(
                f"DualArmController: {side} arm gains set to "
                f"heavy(joints1-4)=({self._ARM_HEAVY_STIFFNESS}, "
                f"{self._ARM_HEAVY_DAMPING}) "
                f"light(joints5-7)=({self._ARM_LIGHT_STIFFNESS}, "
                f"{self._ARM_LIGHT_DAMPING})",
                flush=True,
            )

    def _measured_spine(self) -> float:
        return float(
            self.robot.data.joint_pos[0, self.joint_groups.spine[0]].item()
        )

    def measured_spine_position(self) -> float:
        """Return the live prismatic spine position in metres."""
        return self._measured_spine()

    def commanded_spine_position(self) -> float | None:
        """Return the tracker's current commanded spine target in metres.

        Read-only accessor for M1-V4 logging (plans/handoff.md sec 15.6):
        ``command()`` already solves IK against ``targets.spine`` every tick;
        this exposes that same value so a caller can log measured-vs-commanded
        divergence instead of discarding it.
        """
        if self._tracker is None:
            return None
        return float(self._tracker.targets.spine)

    def sync_targets_from_measured(
        self, *, preserve_gripper: bool = True
    ) -> None:
        """Re-anchor tracker targets after direct-joint transit motions.

        ``preserve_gripper`` (2026-08-21, DOCTOR.md 4.4): carry the
        COMMANDED gripper targets across the rebuild instead of re-anchoring
        them to whatever the jaws are measured at.

        Re-anchoring the arms is the whole point of this method -- a
        direct-joint move (``ramp_arm_pose``) leaves the tracker's Cartesian
        targets stale, so they must be re-read. The gripper is different:
        its commanded value is a DECISION ("hold this width"), not a pose to
        be recovered. Re-reading it destroys that decision exactly when it
        matters most. Under a real load the measured position is not the
        commanded one -- the 2026-08-20 gamepad session measured a genuine
        cup-wall grip at 0.783 rad while commanded fully closed -- so
        re-anchoring silently replaces "squeeze" with "sit at whatever the
        object has wedged the jaws to", and the next disturbance drops it.

        This matters because ``world_isaac.navigate_to()`` calls this
        method (via its pre-drive tuck) on the carry path, while holding.
        ``world_isaac``'s bimanual carry loop already works around it by
        re-commanding the latched width every tick, with a long comment
        attributing the loss to ``set_arm_target_relative``. That
        attribution is wrong -- ``set_arm_target`` goes through
        ``_tracker.apply()``, which preserves the gripper fields. This
        method is the one that drops them.
        """
        from teleop_targets import Pose, TeleopTargets, pose_world_to_base

        root_position, root_orientation = self._root_pose(self.robot)
        spine = self._measured_spine()
        left_world, right_world = self._ik.current_end_effector_poses(
            root_position, root_orientation, spine
        )
        left_relative = pose_world_to_base(
            Pose(tuple(left_world[0]), tuple(left_world[1])),
            root_position,
            root_orientation,
        )
        right_relative = pose_world_to_base(
            Pose(tuple(right_world[0]), tuple(right_world[1])),
            root_position,
            root_orientation,
        )
        if preserve_gripper and self._tracker is not None:
            left_gripper = float(self._tracker.targets.left_gripper)
            right_gripper = float(self._tracker.targets.right_gripper)
        else:
            positions = self.robot.data.joint_pos[0]
            left_gripper = float(
                positions[self.joint_groups.left_gripper[0]].item()
            )
            right_gripper = float(
                positions[self.joint_groups.right_gripper[0]].item()
            )
        self._tracker = self._CartesianTargetTracker(
            TeleopTargets(
                left=left_relative,
                right=right_relative,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                spine=spine,
            ),
            limits=self._TargetLimits(
                position_min=(-1.5, -1.5, -0.5),
                position_max=(1.5, 1.5, 2.5),
                # Read the driven joints' OWN authored travel rather than
                # hardcoding it. `gripper_max` was 1.0 -- past the measured
                # 0.8203 mechanical limit -- so the tracker would happily
                # hold a commanded target the joint can never reach, and a
                # measured 1.0028 (the object levering the jaws open to
                # their stop) read as a legal position rather than an
                # out-of-range one. Same defect class as the 0.9 open
                # constant that cost 28/28 failures on 2026-08-20.
                gripper_min=min(
                    self._gripper_position_lower_limit(side)
                    for side in ("left", "right")
                ),
                gripper_max=max(
                    self._gripper_position_upper_limit(side)
                    for side in ("left", "right")
                ),
                spine_min=0.0,
                spine_max=0.85,
            ),
        )
        self._position_targets = self._measured_position_targets(self.robot)

    @property
    def spine(self) -> float:
        return float(self._tracker.targets.spine)

    @spine.setter
    def spine(self, position_m: float) -> None:
        from teleop_commands import TeleopCommand

        delta = float(position_m) - self.spine
        self._tracker.apply(
            TeleopCommand(
                timestamp=0.0,
                source="task3_autonomy.spine",
                active=True,
                spine_delta=delta,
            )
        )

    def ee_world_poses(self):
        """Measured ``(position, quat_wxyz)`` world pose for each arm."""
        root_position, root_orientation = self._root_pose(self.robot)
        left, right = self._ik.current_end_effector_poses(
            root_position, root_orientation, self._measured_spine()
        )
        return (
            (tuple(left[0]), tuple(left[1])),
            (tuple(right[0]), tuple(right[1])),
        )

    def arm_pose_relative(self, side: str):
        """Return a measured end-effector pose in the robot-base frame."""
        from teleop_targets import Pose, pose_world_to_base

        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        pose = self.ee_world_poses()[0 if side == "left" else 1]
        root_position, root_orientation = self._root_pose(self.robot)
        return pose_world_to_base(
            Pose(tuple(pose[0]), tuple(pose[1])),
            root_position,
            root_orientation,
        )

    def set_arm_target(self, side: str, position, quat_wxyz) -> None:
        """Apply one ``TeleopCommand`` that sets an absolute world target."""
        from teleop_targets import Pose, pose_world_to_base

        world_target = Pose(tuple(position), tuple(quat_wxyz))
        root_position, root_orientation = self._root_pose(self.robot)
        current = getattr(self._tracker.targets, side, None)
        if current is None:
            raise ValueError("side must be 'left' or 'right'")
        command = one_step_reach_command(
            current,
            world_target,
            root_position,
            root_orientation,
            side=side,
        )
        updated = self._tracker.apply(command)
        actual = getattr(updated, side)
        requested = pose_world_to_base(
            world_target, root_position, root_orientation
        )
        position_error = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(actual.position, requested.position)
            )
        )
        orientation_error = _quaternion_angle_error(
            actual.orientation_wxyz, requested.orientation_wxyz
        )
        if position_error > 1.0e-8 or orientation_error > 1.0e-7:
            raise ValueError(
                "world target lies outside CartesianTargetTracker limits"
            )

    def set_arm_target_relative(self, side: str, position, quat_wxyz) -> None:
        """Set an arm target expressed in the current robot-base frame.

        This is the carry counterpart to :meth:`set_arm_target`: a held
        object should move with the robot base rather than leave its gripper
        target fixed in the world while the robot drives through the room.
        """
        from teleop_targets import Pose

        root_position, root_orientation = self._root_pose(self.robot)
        world_target = self._pose_base_to_world(
            Pose(tuple(position), tuple(quat_wxyz)),
            root_position,
            root_orientation,
        )
        self.set_arm_target(
            side, world_target.position, world_target.orientation_wxyz
        )

    def set_gripper(self, side: str, position_rad: float) -> None:
        from teleop_commands import TeleopCommand

        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        field = f"{side}_gripper"
        current = float(getattr(self._tracker.targets, field))
        kwargs = {f"{field}_delta": float(position_rad) - current}
        self._tracker.apply(
            TeleopCommand(
                timestamp=0.0,
                source="task3_autonomy.gripper",
                active=True,
                **kwargs,
            )
        )
        # DEBUG (owner-requested verification, not permanent): confirm
        # set_gripper's INPUT actually lands as the tracker's TARGET.
        if side == "right":
            print(
                f"GRIPPER_SET_DEBUG side={side} input_rad={float(position_rad):.5f} "
                f"tracker_target={float(getattr(self._tracker.targets, field)):.5f}",
                flush=True,
            )

    def gripper_position(self, side: str) -> float:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        ids = getattr(self.joint_groups, f"{side}_gripper")
        return float(self.robot.data.joint_pos[0, ids[0]].item())

    def _gripper_effort_limit(self, side: str) -> float:
        """Return the authored effort limit for a primary gripper joint."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        limits = getattr(self.robot.data, "joint_effort_limits", None)
        if limits is None:
            raise RuntimeError("robot does not expose joint effort limits")
        joint_id = getattr(self.joint_groups, f"{side}_gripper")[0]
        limit = float(limits[0, joint_id].item())
        if not math.isfinite(limit) or limit <= 0.0:
            raise RuntimeError(
                "gripper authored effort limit must be positive"
            )
        return limit

    def _gripper_position_upper_limit(self, side: str) -> float:
        """Return the authored upper position limit for a primary gripper joint.

        `DEFAULT_GRIPPER_OPEN_RAD` is a fallback for callers with no live
        robot. The value it replaced (1.05) was a hand-picked number that
        sat past the USD's authored 0..1 rad joint limit, so a gripper jammed
        fully open (measured 1.0145 rad -- past its own mechanical limit
        under contact-induced overshoot) still passed `min < position < max`
        and was scored as a successful hold. The joint's own authored limit
        is the correct ceiling: reading it here instead of hardcoding a
        second number keeps the two in sync automatically.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        limits = getattr(self.robot.data, "joint_pos_limits", None)
        if limits is None:
            raise RuntimeError("robot does not expose joint position limits")
        joint_id = getattr(self.joint_groups, f"{side}_gripper")[0]
        upper = float(limits[0, joint_id, 1].item())
        if not math.isfinite(upper) or upper <= 0.0:
            raise RuntimeError(
                "gripper authored upper position limit must be positive"
            )
        return upper

    def _gripper_position_lower_limit(self, side: str) -> float:
        """Return the authored LOWER position limit (fully closed).

        Sibling of the upper-limit reader above, for the same reason: the
        closed end was hardcoded as 0.0 in the tracker's limits, and a
        per-side or per-asset difference would be silently clamped away.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        limits = getattr(self.robot.data, "joint_pos_limits", None)
        if limits is None:
            raise RuntimeError("robot does not expose joint position limits")
        joint_id = getattr(self.joint_groups, f"{side}_gripper")[0]
        lower = float(limits[0, joint_id, 0].item())
        if not math.isfinite(lower):
            raise RuntimeError(
                "gripper authored lower position limit must be finite"
            )
        return lower

    def set_gripper_effort_scale(self, side: str, scale: float) -> None:
        """Scale the gripper's authored maximum effort for physical closure."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("effort scale must be finite and in (0, 1]")
        joint_ids = getattr(self.joint_groups, f"{side}_gripper")
        self.robot.write_joint_effort_limit_to_sim(
            self._default_gripper_effort_limits[side] * scale,
            joint_ids=joint_ids,
        )

    def set_gripper_stiffness(self, side: str, stiffness: float) -> None:
        """Set the driven gripper joint's position gain.

        Nothing in this codebase ever set this, so the gripper has always run
        on the asset's authored stiffness of 3.0. That is the binding
        constraint on closing, not the effort limit: torque is
        `stiffness * error`, so even a full 1.0 rad of error produces only
        ~3 N*m against an authored effort limit of 50 N*m -- the limit is
        never reached and `set_gripper_effort_scale` (which can only scale
        DOWN, 0 < scale <= 1) cannot help.

        Measured, aspire_1: the close commands 0 for 300 ticks and the joint
        does not move at all, sitting at 1.0039 rad -- 0.1 rad PAST the 0.9 it
        was opened to, i.e. wedged further open by contact. `contact_sustained`
        there means the error never fell, which is what a stuck joint looks
        like.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if not math.isfinite(stiffness) or stiffness <= 0.0:
            raise ValueError("stiffness must be finite and positive")
        import torch

        joint_ids = list(getattr(self.joint_groups, f"{side}_gripper"))
        self.robot.write_joint_stiffness_to_sim(
            torch.full(
                (1, len(joint_ids)),
                float(stiffness),
                device=self.robot.data.joint_pos.device,
            ),
            joint_ids=joint_ids,
        )

    def restore_gripper_effort_limit(self, side: str) -> None:
        """Restore the gripper's authored maximum effort limit."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        joint_ids = getattr(self.joint_groups, f"{side}_gripper")
        self.robot.write_joint_effort_limit_to_sim(
            self._default_gripper_effort_limits[side],
            joint_ids=joint_ids,
        )

    def preview_arm_joints(
        self,
        side: str,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
    ) -> list[float] | None:
        """Joint angles IK would choose for a hypothetical target, WITHOUT
        commanding anything.

        Exists so a caller can compare two candidate end-effector poses by
        the joint motion they actually require, rather than by how far apart
        the poses look. Those are different questions: this arm is redundant,
        so Lula picks a solution out of a null space and two nearby EE
        orientations can demand very different wrist angles. Measured on
        run 14 -- choosing between a grasp roll and its 180-degree twin by
        quaternion distance changed `recenter_pos_err_m` by 0.001, i.e.
        nothing, because the roll requested was never what set joint 7's
        travel.

        The other arm is pinned at its current measured pose so the solve
        answers "what would THIS arm do", and returns None if IK finds
        nothing.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        root_position, root_orientation = self._root_pose(self.robot)
        spine = self._measured_spine()
        left_now, right_now = self._ik.current_end_effector_poses(
            root_position, root_orientation, spine
        )
        if side == "left":
            left_p, left_q = tuple(position), tuple(orientation_wxyz)
            right_p, right_q = tuple(right_now[0]), tuple(right_now[1])
        else:
            right_p, right_q = tuple(position), tuple(orientation_wxyz)
            left_p, left_q = tuple(left_now[0]), tuple(left_now[1])
        result = self._ik.solve(
            left_p,
            right_p,
            left_q,
            right_q,
            spine_position=spine,
            base_position=root_position,
            base_orientation_wxyz=root_orientation,
        )
        names = (
            self._left_arm_joint_names
            if side == "left"
            else self._right_arm_joint_names
        )
        solution = result.left if side == "left" else result.right
        try:
            return ordered_joint_targets(solution, names)
        except ValueError:
            return None

    def follow_arm_joint_trajectory(
        self,
        side: str,
        waypoints,
        *,
        step,
        ticks_per_waypoint: int = 1,
        settle_ticks: int = 0,
    ) -> None:
        """Drive one arm through a planned joint trajectory.

        This is the point of the whole cuMotion exercise. `reach()` commands
        a Cartesian target and lets the IK-servo chase it, which measurably
        ends 0.05-0.07 m short with IK solving on every tick. A planned
        trajectory is a sequence of joint states that is already feasible and
        already ENDS at the solution, so following it lands where the planner
        said rather than wherever the servo converged.

        Writes the arm's joints directly rather than going through the
        tracker: the tracker exists to turn Cartesian deltas into IK
        solutions, and there is nothing to solve here.
        """
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        arm_ids = list(getattr(self.joint_groups, f"{side}_arm"))
        for waypoint in waypoints:
            values = [float(v) for v in waypoint][: len(arm_ids)]
            if len(values) < len(arm_ids):
                continue
            target = self._position_targets.new_tensor([values])
            self.robot.set_joint_position_target(target, joint_ids=arm_ids)
            # Keep the tracker's own record in step so a later Cartesian
            # command starts from where the trajectory actually left the arm,
            # not from a stale target.
            self._position_targets[0, arm_ids] = target[0]
            for _ in range(max(1, ticks_per_waypoint)):
                step()
        for _ in range(max(0, settle_ticks)):
            step()

    def measured_arm_joints(self, side: str) -> list[float]:
        """Current measured joint angles for one arm, in composer order."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        ids = list(getattr(self.joint_groups, f"{side}_arm"))
        return [float(self.robot.data.joint_pos[0, i]) for i in ids]

    def command(self):
        """Solve the current tracker targets and write articulation targets."""
        ik_result, composed = self.solve_position_targets()
        self.commit_position_targets(composed)
        return ik_result

    def solve_position_targets(self):
        """Solve IK and COMPOSE the next articulation targets, WITHOUT writing.

        Split out of ``command()`` (2026-08-21) so a caller can inspect what
        the solver just produced *before* the robot is told to execute it.

        This is what makes ``reach()``'s joint-thrash guard preventive
        instead of forensic. The loop used to be
        ``set_arm_target -> command() -> step() -> check delta``: by the
        time the guard saw a 3.6 rad single-tick flip, ``command()`` had
        already written it and ``step()`` had already executed it. Measured
        (``pinbase_yaw_fix_run3_20260821_040606``, ticks 6173-6280): the
        ``joint_thrash_bailed`` event fired on the SAME tick the cup was
        flung from ``[-4.183,-1.752,0.751]`` to ``[-5.619,-1.246,0.034]``
        -- off the table onto the floor. The guard correctly reported an
        accident it was positioned too late to prevent.

        Returns ``(ik_result, composed_targets)``. Nothing is mutated, so a
        rejected solution can simply be dropped: the articulation keeps
        whatever target was last committed, which is the same "freeze
        here" contract ``zero_success_bail_ticks`` already relies on.
        """
        targets = self._tracker.targets
        root_position, root_orientation = self._root_pose(self.robot)
        left_world = self._pose_base_to_world(
            targets.left, root_position, root_orientation
        )
        right_world = self._pose_base_to_world(
            targets.right, root_position, root_orientation
        )
        ik_result = self._ik.solve(
            left_world.position,
            right_world.position,
            left_world.orientation_wxyz,
            right_world.orientation_wxyz,
            spine_position=targets.spine,
            base_position=root_position,
            base_orientation_wxyz=root_orientation,
        )
        left_arm = ordered_joint_targets(
            ik_result.left, self._left_arm_joint_names
        )
        right_arm = ordered_joint_targets(
            ik_result.right, self._right_arm_joint_names
        )
        composed = self._compose(
            self._position_targets,
            self.joint_groups,
            left_arm=left_arm,
            right_arm=right_arm,
            left_gripper=targets.left_gripper,
            right_gripper=targets.right_gripper,
            spine=targets.spine,
        )
        return ik_result, composed

    def commit_position_targets(self, composed) -> None:
        """Record and WRITE a solution produced by solve_position_targets."""
        self._position_targets = composed
        position_targets, joint_ids = self._subset(
            composed, self.joint_groups
        )
        self.robot.set_joint_position_target(
            position_targets, joint_ids=joint_ids
        )

    def pose_error(
        self, side: str, position, quat_wxyz
    ) -> tuple[float, float]:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        measured = self.ee_world_poses()[0 if side == "left" else 1]
        position_error = math.sqrt(
            sum((m - t) ** 2 for m, t in zip(measured[0], position))
        )
        return position_error, _quaternion_angle_error(measured[1], quat_wxyz)

    def position_error(self, side: str, target_position) -> float:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        measured = self.ee_world_poses()[0 if side == "left" else 1][0]
        return math.sqrt(
            sum((m - t) ** 2 for m, t in zip(measured, target_position))
        )

    def commanded_arm_joint_positions(self, side: str) -> list[float]:
        """The current COMMANDED (not measured) joint targets for one arm.

        2026-08-09 (O1 investigation): extracted from `reach()`'s own
        internal joint-thrash check so callers with their OWN raw
        `set_arm_target`/`command()` tick loop (e.g.
        `world_isaac.py::_push_object_to`'s `push_gentle_ramp`, which
        does not go through `reach()`) can apply the same tick-to-tick
        joint-delta guard without duplicating the indexing logic.
        """
        arm_joint_ids = list(getattr(self.joint_groups, f"{side}_arm"))
        return self._position_targets[0, arm_joint_ids].tolist()

    def reach(
        self,
        side: str,
        position,
        quat_wxyz,
        *,
        step: Callable[[], None],
        dt: float,
        timeout_s: float,
        position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = DEFAULT_ORIENTATION_TOLERANCE_RAD,
        ik_stats: dict[str, Any] | None = None,
        zero_success_bail_ticks: int | None = None,
        max_joint_delta_rad: float | None = None,
        plateau_bail_ticks: int | None = None,
        plateau_min_progress_m: float = 0.005,
    ) -> bool:
        """Reach a world pose, returning False on explicit timeout.

        If ``ik_stats`` is passed, it is filled in-place with per-tick IK
        diagnostics (``ticks``, ``ik_ok_ticks``, ``ik_fail_ticks``,
        ``first_ik_fail_tick``, ``ee_pos_final``, ``per_axis_err_m``) so a
        caller can tell an IK-frozen arm (stale joints on solver failure,
        see ``dual_arm_lula._solve_arm``) apart from an arm that converges
        but is physically blocked from the target.

        ``max_joint_delta_rad`` (2026-08-09, O1 investigation): `None`
        (default) is the ORIGINAL, unaffected behavior at every existing
        call site. When given, bails immediately (same "freeze here"
        contract as `zero_success_bail_ticks`) the first tick the
        COMMANDED joint target jumps by more than this amount (any single
        joint) from the previous tick's commanded target -- GPU-confirmed
        real hazard, already documented but never fixed
        (`docker/SUBMISSION_README.md`'s own "Known limitations": a
        `reach()` call that reports IK success every tick can still
        internally flip between different valid joint-space solutions
        mid-servo, each `succeeded=True`, with no existing guard (the
        `zero_success_bail_ticks` protection only watches for IK
        *failure*, not "succeeding every tick while thrashing"). A single
        such flip sweeps the arm through a large, physically real
        excursion in one tick regardless of tolerance checks --
        `spoon2_run10_seed7_contactfallback.log`'s own `push_approach`
        attempt shows exactly this: `joint_tracking_trace` commanded
        values jump from `[-2.19, -1.80, 2.84, ...]` to
        `[0.70, 1.73, -0.06, ...]` between two 200-tick samples, and the
        object was flung ~1.4m onto the floor in that same window. Opt-in
        only, not enabled at any call site until GPU-verified.

        ``zero_success_bail_ticks``, if given, returns False early once
        that many CONSECUTIVE ticks have failed (handoff.md sec 78/80: a
        target the solver has not found ANY solution for after this many
        ticks is a much stronger unreachability signal than "not yet
        converged" -- a genuinely reachable target starts succeeding
        almost immediately, per the same log's push_approach attempts, even
        when it still needs the full budget to shrink its error under
        tolerance). On IK failure ``dual_arm_lula._solve_arm`` freezes the
        commanded joint target at the last successful solution rather than
        moving erratically, so the caller holds a fixed, possibly
        interpenetrating pose for whatever budget is left -- bailing early
        shortens that hold instead of grinding out the full timeout next to
        the object. Off by default (``None``) so every existing caller is
        unaffected.

        handoff.md sec 96: originally this only bailed while
        ``ik_ok_ticks`` stayed at 0 for the whole call, so a target that
        succeeded even once (however briefly) lost this protection for
        the rest of the budget -- observed live flinging an object ~5.5 m
        after 417 successful ticks were followed by 783 straight
        failures. Tracking CONSECUTIVE failures (reset on any success)
        instead closes that gap while still covering the original
        never-succeeded case (there, consecutive failures equal total
        ticks).

        ``plateau_bail_ticks`` (2026-08-20, EBiM Task 3 cup-grasp sprint):
        the other three guards above all catch different flavors of
        *danger* (thrash, runaway, never-solving) but none catch plain
        *stagnation* -- IK solving fine, no large joint jump, error just
        stuck a few mm above ``position_tolerance_m`` for the rest of the
        budget. Measured real (``plans/PROGRESS.md`` 2026-08-20): pregrasp
        calls converge to ~0.02-0.04m within ~200 ticks then idle ~1400
        more, ~40 real minutes/run of nothing. Same rolling-window-trend
        shape as ``JOINT_RUNAWAY_TREND_TICKS`` below, just watching
        ``position_error`` instead of joint delta: bail once the window is
        full AND the error hasn't shrunk by at least
        ``plateau_min_progress_m`` across it. Off by default -- every
        existing caller is unaffected until it opts in.
        """
        if dt <= 0.0 or timeout_s < 0.0:
            raise ValueError("dt must be positive and timeout_s non-negative")
        if (
            zero_success_bail_ticks is not None
            and zero_success_bail_ticks <= 0
        ):
            raise ValueError("zero_success_bail_ticks must be positive")
        ticks = 0
        ik_ok_ticks = 0
        ik_fail_ticks = 0
        first_ik_fail_tick: int | None = None
        consecutive_fail_ticks = 0
        succeeded = False
        # handoff sec 97: a 100%-IK-solving push_contact attempt still
        # plateaued ~0.13m short of the target -- only the FINAL tick's
        # error was ever visible, so it was impossible to tell whether
        # that was a slow-but-real convergence (needs more ticks) or a
        # stuck offset (needs a different fix). Sample coarsely (every 200
        # ticks) so the next real attempt answers that question for free.
        position_error_trace: list[tuple[int, float]] = []
        # VM_A_BRIEF A1: position_error_trace shows the plateau but not
        # whether the drive is even tracking what it was told -- an
        # untuned/over-damped PD gain would show commanded reaching the
        # target while actual lags behind and never closes. Same sampling
        # cadence as position_error_trace so the two traces line up tick
        # for tick.
        arm_joint_ids = list(getattr(self.joint_groups, f"{side}_arm"))
        joint_tracking_trace: list[tuple[int, list[float], list[float]]] = []
        prev_commanded_joints: list[float] | None = None
        joint_thrash_tick: int | None = None
        joint_thrash_delta_rad: float | None = None
        # Owner-directed safety guard (2026-08-19): the existing thrash
        # check above only compares COMMANDED targets tick-to-tick, so it
        # never catches the physical joint itself running away while the
        # commanded target stays sane. Measured live (GPU run,
        # outputs/task3_cup_teleport_no_nav/run.log): joint index 4 (left
        # arm joint 5) reached a MEASURED 37.0 rad while its commanded
        # target stayed near -2.1 rad for hundreds of consecutive ticks --
        # a real drive/tracking failure, not an IK or reachability
        # problem -- and the resulting violent motion swept the cup AND
        # the plate off the table (video-confirmed). Root cause not yet
        # found; this is a protective circuit-breaker so a runaway joint
        # can no longer destroy the scene while that gets investigated.
        # 1.0 rad is well past the ~0.2-0.4 rad tracking lag this
        # codebase's own prior GPU runs have measured as normal.
        JOINT_RUNAWAY_THRESHOLD_RAD = 1.0
        joint_runaway_tick: int | None = None
        joint_runaway_delta_rad: float | None = None
        _runaway_history: list[tuple[int, float]] = []
        plateau_tick: int | None = None
        plateau_error_m: float | None = None
        _plateau_history: list[tuple[int, float]] = []
        for tick in range(math.ceil(timeout_s / dt)):
            # Tracker poses are base-relative. Reissue the absolute target
            # every tick so base reaction/drift cannot carry the goal away.
            self.set_arm_target(side, position, quat_wxyz)
            # SOLVE, INSPECT, THEN COMMIT -- in that order. `command()`
            # solved and wrote in one call, so the thrash check below could
            # only ever run on a target the robot had already executed.
            # See `solve_position_targets` for the measured cup-flinging
            # this ordering exists to prevent.
            ik_result, _candidate_targets = self.solve_position_targets()
            commanded_now = _candidate_targets[0, arm_joint_ids].tolist()
            if (
                max_joint_delta_rad is not None
                and prev_commanded_joints is not None
            ):
                delta = max(
                    abs(c - p)
                    for c, p in zip(commanded_now, prev_commanded_joints)
                )
                if delta > max_joint_delta_rad:
                    # DISCARD the solution: never written, never stepped.
                    # The articulation holds its last committed target, so
                    # the arm freezes where it was rather than swinging
                    # through the flip and then being told to stop.
                    joint_thrash_tick = tick
                    joint_thrash_delta_rad = round(delta, 4)
                    break
            prev_commanded_joints = commanded_now
            self.commit_position_targets(_candidate_targets)
            step()
            position_error, orientation_error = self.pose_error(
                side, position, quat_wxyz
            )
            succeeded = (
                ik_result.left_succeeded
                if side == "left"
                else ik_result.right_succeeded
            )
            ticks += 1
            if succeeded:
                ik_ok_ticks += 1
                consecutive_fail_ticks = 0
            else:
                ik_fail_ticks += 1
                consecutive_fail_ticks += 1
                if first_ik_fail_tick is None:
                    first_ik_fail_tick = tick
            # The thrash guard ran BEFORE the commit above (2026-08-09's
            # every-tick commanded-target check, moved 2026-08-21 so it can
            # reject a flip instead of reporting one). `commanded_now` is
            # reused here rather than re-read: `.tolist()` on a CUDA tensor
            # forces a device synchronization -- the CPU blocks until every
            # queued kernel finishes -- and this loop used to pay that
            # twice per tick for the same slice. At dt=0.005 every
            # redundant sync costs 200x per simulated second, against a
            # measured 86-107 ms of WALL time per tick already
            # (`WORLD_ISAAC_TICK`'s own `s_per_tick`, runs 1 and 3,
            # 2026-08-21) and a physics step that should be ~1-2 ms.
            #
            # Unconditional (unlike the thrash check, which only runs when
            # a caller opts in via max_joint_delta_rad) -- a runaway
            # physical joint is dangerous regardless of whether that
            # separate, optional feature is enabled for this call.
            actual_now = self.robot.data.joint_pos[0, arm_joint_ids].tolist()
            runaway_delta = max(
                abs(c - a) for c, a in zip(commanded_now, actual_now)
            )
            # Tick 0 of a fresh target legitimately starts far from
            # commanded (the arm hasn't moved yet) -- that is normal PD
            # lag, not a runaway. GPU-confirmed false-positive: this fired
            # on every phase's very first tick before the settle grace
            # below was added. The real bug this guards against grows
            # over hundreds of ticks while commanded stays fixed (measured
            # GPU trace: 21.7 -> 27.7 -> 33.6 -> 37.0 rad across ~1000+
            # ticks) -- a bounded settle window does not mask that.
            # Redesigned 2026-08-19 (GPU-confirmed TWICE that a fixed tick
            # count cannot get this right in both directions): a settle
            # window of 100 caught a real runaway too late (2.385 rad
            # accumulated by tick 101, cup knocked off); tightening it to
            # 20 then broke every LEGITIMATE large descend (a real 0.2-
            # 0.5m standoff/grasp_height move needs far more than 20
            # ticks just to start converging -- ee_final got stuck ~0.3-
            # 0.5m short of target on every attempt, ik_ok_ticks exactly
            # matching the tick count each time because THIS check was
            # cutting the loop at tick ~20-25 before real motion had a
            # chance to happen). A fixed threshold cannot distinguish
            # "large gap, shrinking" (normal PD convergence toward a big
            # intentional move) from "gap not shrinking" (a genuine
            # runaway/stall) -- so track the trend instead of a single
            # snapshot: keep a short rolling window of past deltas, and
            # only bail when the gap is BOTH over threshold AND has not
            # decreased across that window (still stuck/diverging, not
            # just slow).
            JOINT_RUNAWAY_TREND_TICKS = 60
            if runaway_delta > JOINT_RUNAWAY_THRESHOLD_RAD:
                _runaway_history.append((tick, runaway_delta))
                while (
                    _runaway_history
                    and tick - _runaway_history[0][0] > JOINT_RUNAWAY_TREND_TICKS
                ):
                    _runaway_history.pop(0)
                oldest_delta = _runaway_history[0][1]
                stuck_long_enough = (
                    tick - _runaway_history[0][0] >= JOINT_RUNAWAY_TREND_TICKS
                )
                if stuck_long_enough and runaway_delta >= oldest_delta:
                    joint_runaway_tick = tick
                    joint_runaway_delta_rad = round(runaway_delta, 4)
                    break
            else:
                _runaway_history.clear()
            if plateau_bail_ticks is not None and not (
                succeeded and position_error <= position_tolerance_m
            ):
                # Same rolling-window-trend shape as JOINT_RUNAWAY above,
                # watching position_error instead of joint delta: only
                # fires once the window is genuinely full (never on a
                # target that converges quickly) AND the error hasn't
                # meaningfully shrunk across it (never on ordinary
                # convergence, which shrinks the window-start error every
                # tick).
                _plateau_history.append((tick, position_error))
                while (
                    _plateau_history
                    and tick - _plateau_history[0][0] > plateau_bail_ticks
                ):
                    _plateau_history.pop(0)
                if tick - _plateau_history[0][0] >= plateau_bail_ticks:
                    oldest_error = _plateau_history[0][1]
                    if oldest_error - position_error < plateau_min_progress_m:
                        plateau_tick = tick
                        plateau_error_m = round(position_error, 4)
                        break
            if ticks == 1 or ticks % 200 == 0:
                # orientation_error alongside position_error, in degrees.
                # 2026-08-15: these two traces disagreed and nothing recorded
                # enough to say which was lying. `descend_standoff` returned
                # ok=True -- which requires orientation_error <= 5 deg -- in
                # a run whose joint_tracking_trace showed joint 7, the
                # TERMINAL wrist roll, sitting 0.4168 rad (24 deg) from its
                # commanded value with joint 6 only 0.096 off. Those cannot
                # both be true, and the answer decides whether the arm has a
                # drive-rate problem at all or whether the joint delta is
                # just tracker interpolation lag being misread.
                position_error_trace.append(
                    (
                        ticks,
                        round(position_error, 4),
                        round(math.degrees(orientation_error), 2),
                    )
                )
                commanded = self._position_targets[0, arm_joint_ids].tolist()
                actual = self.robot.data.joint_pos[0, arm_joint_ids].tolist()
                joint_tracking_trace.append(
                    (
                        ticks,
                        [round(v, 4) for v in commanded],
                        [round(v, 4) for v in actual],
                    )
                )
            if (
                succeeded
                and position_error <= position_tolerance_m
                and orientation_error <= orientation_tolerance_rad
            ):
                if ik_stats is not None:
                    self._fill_ik_stats(
                        ik_stats,
                        side,
                        position,
                        ticks,
                        ik_ok_ticks,
                        ik_fail_ticks,
                        first_ik_fail_tick,
                        position_error_trace,
                        joint_tracking_trace,
                    )
                return True
            if (
                zero_success_bail_ticks is not None
                and consecutive_fail_ticks >= zero_success_bail_ticks
            ):
                break
        position_error, orientation_error = self.pose_error(
            side, position, quat_wxyz
        )
        if ik_stats is not None:
            self._fill_ik_stats(
                ik_stats,
                side,
                position,
                ticks,
                ik_ok_ticks,
                ik_fail_ticks,
                first_ik_fail_tick,
                position_error_trace,
                joint_tracking_trace,
            )
            if joint_thrash_tick is not None:
                ik_stats["joint_thrash_bailed"] = True
                ik_stats["joint_thrash_tick"] = joint_thrash_tick
                ik_stats["joint_thrash_delta_rad"] = joint_thrash_delta_rad
            if joint_runaway_tick is not None:
                ik_stats["joint_runaway_bailed"] = True
                ik_stats["joint_runaway_tick"] = joint_runaway_tick
                ik_stats["joint_runaway_delta_rad"] = joint_runaway_delta_rad
            if plateau_tick is not None:
                ik_stats["plateau_bailed"] = True
                ik_stats["plateau_tick"] = plateau_tick
                ik_stats["plateau_error_m"] = plateau_error_m
        if (
            joint_thrash_tick is not None
            or joint_runaway_tick is not None
            or plateau_tick is not None
        ):
            return False
        return (
            position_error <= position_tolerance_m
            and orientation_error <= orientation_tolerance_rad
        )

    def _fill_ik_stats(
        self,
        ik_stats: dict[str, Any],
        side: str,
        target_position,
        ticks: int,
        ik_ok_ticks: int,
        ik_fail_ticks: int,
        first_ik_fail_tick: int | None,
        position_error_trace: list[tuple[int, float]] | None = None,
        joint_tracking_trace: list[tuple[int, list[float], list[float]]]
        | None = None,
    ) -> None:
        ee_pos_final = self.ee_world_poses()[0 if side == "left" else 1][0]
        ik_stats.update(
            ticks=ticks,
            ik_ok_ticks=ik_ok_ticks,
            ik_fail_ticks=ik_fail_ticks,
            first_ik_fail_tick=first_ik_fail_tick,
            ee_pos_final=[round(v, 4) for v in ee_pos_final],
            per_axis_err_m=[
                round(a - b, 4) for a, b in zip(ee_pos_final, target_position)
            ],
        )
        if position_error_trace is not None:
            ik_stats["position_error_trace"] = position_error_trace
        if joint_tracking_trace is not None:
            ik_stats["joint_tracking_trace"] = joint_tracking_trace

    def grasp(
        self,
        side: str,
        *,
        step: Callable[[], None],
        dt: float,
        settle_seconds: float = 1.5,
        ramp_seconds: float = 1.0,
        close_effort_scale: float | None = None,
        hold_target_on_contact: bool = False,
        contact_freeze_max_target_rad: float = (
            DEFAULT_CONTACT_FREEZE_MAX_TARGET_RAD
        ),
        stall_lookback_ticks: int = 1,
        telemetry: dict[str, Any] | None = None,
    ) -> bool:
        """Soft-close, settle, then confirm an object blocks full closure.

        REV13 T2: the tick loop itself lives in the free function
        `run_gripper_close_ramp` so it is CPU-testable without a real
        robot -- pass `telemetry` (a dict) to get the per-tick commanded
        target, measured position, contact tick, final residual, and an
        `outcome` classification (`closed_no_contact` /
        `contact_sustained` / `contact_lost`) populated in place.

        REV13 T4 + T4-followup: `hold_target_on_contact=True` freezes the
        commanded target at first contact instead of continuing to ramp
        toward fully closed, but only when that contact is detected at or
        below `contact_freeze_max_target_rad` -- see
        `run_gripper_close_ramp`'s docstring for the real GPU traces (T3's
        genuine catch vs T4's too-early false freeze) this is grounded in.
        """
        if (
            dt <= 0.0
            or settle_seconds < 0.0
            or ramp_seconds < 0.0
            or ramp_seconds > settle_seconds
        ):
            raise ValueError(
                "dt must be positive and 0 <= ramp_seconds <= settle_seconds"
            )
        start_position = self.gripper_position(side)
        if close_effort_scale is not None:
            self.set_gripper_effort_scale(side, close_effort_scale)
        ramp_ticks = max(1, math.ceil(ramp_seconds / dt))
        total_ticks = math.ceil(settle_seconds / dt)

        def _advance() -> None:
            self.command()
            step()

        return run_gripper_close_ramp(
            start_position=start_position,
            end_position=GRIPPER_CLOSED_RAD,
            total_ticks=total_ticks,
            ramp_ticks=ramp_ticks,
            set_target=lambda target: self.set_gripper(side, target),
            measured_position=lambda: self.gripper_position(side),
            advance=_advance,
            hold_target_on_contact=hold_target_on_contact,
            contact_freeze_max_target_rad=contact_freeze_max_target_rad,
            stall_lookback_ticks=stall_lookback_ticks,
            hold_max_position_rad=self._gripper_position_upper_limit(side),
            telemetry=telemetry,
        )

    def release(
        self,
        side: str,
        *,
        step: Callable[[], None],
        dt: float,
        timeout_s: float = 1.5,
        tolerance_rad: float = 0.02,
    ) -> bool:
        """Open the gripper and return False if it misses the timeout.

        Targets the joint's OWN authored upper limit rather than a module
        constant. This is the exact bug class that cost 28/28 failures on
        2026-08-20 -- `GRIPPER_OPEN_RAD` was 0.9 against an authored limit
        of 0.8203, so `abs(position - target) <= tolerance` could never be
        satisfied no matter how the stiffness or timeout was tuned. Reading
        the limit keeps the two in sync automatically instead of leaving a
        second number to drift, and covers a per-side or per-asset
        difference the shared constant cannot express.
        """
        if dt <= 0.0 or timeout_s < 0.0:
            raise ValueError("dt must be positive and timeout_s non-negative")
        self.restore_gripper_effort_limit(side)
        open_rad = self._gripper_position_upper_limit(side)
        self.set_gripper(side, open_rad)

        def _open_enough() -> bool:
            return abs(self.gripper_position(side) - open_rad) <= tolerance_rad

        for _ in range(math.ceil(timeout_s / dt)):
            self.command()
            step()
            if _open_enough():
                return True
        return _open_enough()

    def move_spine(
        self,
        position_m: float,
        *,
        step: Callable[[], None],
        dt: float,
        timeout_s: float = 4.0,
        tolerance_m: float = 0.01,
    ) -> bool:
        """Move the spine with measured convergence and an explicit timeout."""
        if dt <= 0.0 or timeout_s < 0.0:
            raise ValueError("dt must be positive and timeout_s non-negative")
        self.spine = position_m
        for _ in range(math.ceil(timeout_s / dt)):
            self.command()
            step()
            if abs(self._measured_spine() - position_m) <= tolerance_m:
                return True
        return abs(self._measured_spine() - position_m) <= tolerance_m

    def lift(
        self,
        side: str,
        dz: float,
        *,
        step: Callable[[], None],
        dt: float,
        timeout_s: float,
        position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M,
        ramp_seconds: float = 3.0,
        spine_assist_m: float = 0.0,
        on_tick: Callable[[int], None] | None = None,
    ) -> bool:
        """Raise vertically with a bounded ramp while holding attitude.

        A full-height position step can accelerate a pinched object sideways
        before the fingers have developed a stable contact.  The ramp keeps
        every intermediate IK request close to the measured configuration and
        leaves the remainder of ``timeout_s`` for convergence at full height.

        ``on_tick``, if given, is called with the tick index right after
        ``step()`` every tick -- REV13 T4-followup-2's finding that lift
        fails even after a telemetrically "sustained" grasp means the lift
        motion itself needs the same per-tick visibility T2 gave the close
        ramp. Callers with access to the held object (``world_isaac.py``)
        use this to sample object-vs-EE tracking during the lift, not just
        the pose-convergence math this method already sees.
        """
        if (
            dt <= 0.0
            or timeout_s < 0.0
            or ramp_seconds < 0.0
            or ramp_seconds > timeout_s
            or spine_assist_m < 0.0
        ):
            raise ValueError(
                "dt must be positive, 0 <= ramp_seconds <= timeout_s, "
                "and spine_assist_m non-negative"
            )
        measured = self.ee_world_poses()[0 if side == "left" else 1]
        start_spine = self.spine
        start_position = measured[0]
        final_position = (
            start_position[0],
            start_position[1],
            start_position[2] + float(dz),
        )
        ramp_ticks = max(1, math.ceil(ramp_seconds / dt))
        timeout_ticks = math.ceil(timeout_s / dt)
        for tick in range(timeout_ticks):
            completed_steps = tick + 1
            target = (
                start_position[0],
                start_position[1],
                linear_ramp_target(
                    start_position[2],
                    final_position[2],
                    completed_steps,
                    ramp_ticks,
                ),
            )
            if spine_assist_m > 0.0:
                self.spine = linear_ramp_target(
                    start_spine,
                    start_spine + spine_assist_m,
                    completed_steps,
                    ramp_ticks,
                )
            self.set_arm_target(side, target, measured[1])
            ik_result = self.command()
            step()
            if on_tick is not None:
                on_tick(tick)
            if tick + 1 < ramp_ticks:
                continue
            position_error, orientation_error = self.pose_error(
                side, final_position, measured[1]
            )
            succeeded = (
                ik_result.left_succeeded
                if side == "left"
                else ik_result.right_succeeded
            )
            if (
                succeeded
                and position_error <= position_tolerance_m
                and orientation_error <= DEFAULT_ORIENTATION_TOLERANCE_RAD
            ):
                return True
        position_error, orientation_error = self.pose_error(
            side, final_position, measured[1]
        )
        return (
            position_error <= position_tolerance_m
            and orientation_error <= DEFAULT_ORIENTATION_TOLERANCE_RAD
        )

    def place(self, side: str, position, quat_wxyz, **kwargs) -> bool:
        """Place is the pose-convergent motion portion of reach→release."""
        return self.reach(side, position, quat_wxyz, **kwargs)
