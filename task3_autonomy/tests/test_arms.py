# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the autonomous reach command and grasp predicate."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "common"))

from teleop_targets import (  # noqa: E402
    CartesianTargetTracker,
    Pose,
    TargetLimits,
    TeleopTargets,
    _quaternion_from_rpy,
    pose_world_to_base,
)

from task3_autonomy.arms import (  # noqa: E402
    DEFAULT_CONTACT_ERROR_RAD,
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    DualArmController,
    grasp_lift_gate_passed,
    gripper_holds_object,
    linear_ramp_target,
    one_step_reach_command,
    ordered_joint_targets,
    synchronized_drag_targets,
)


def test_changingtek_gripper_uses_zero_closed_convention():
    assert GRIPPER_CLOSED_RAD == 0.0
    # 2026-08-20: was 0.9, past this asset's own USD-authored joint limit
    # (measured directly, scripts/task3/probe_gripper_joint_limits.py:
    # [0.0, 0.8203047513961792] both sides) -- 0.9 was never reachable,
    # which is why every `release()`/`open_before_approach` call failed.
    assert GRIPPER_OPEN_RAD == 0.82


def test_configure_arm_joint_gains_scoped_to_configured_sides_only():
    """T2 (SYNC 33): the override must only touch the side(s) named in
    `_ARM_GAIN_OVERRIDE_SIDES` -- SYNC 35 found an override unconditional
    on both arms broke Stage 1 pregrasp by ~0.2m.

    Isolated from `_ARM_LIGHT_DAMPING_ONLY_SIDES` (monkeypatched to empty
    here): that is a separate, independently-scoped damping-only
    mechanism (see its own class docstring) that also calls
    `write_joint_damping_to_sim`, and 8939f3a found leaving both enabled
    made this test's `touched_ids == expected_ids` assertion fail even
    though `_ARM_GAIN_OVERRIDE_SIDES` scoping itself was untouched -- a
    test-isolation gap, not evidence the light-damping mechanism is
    unsafe. `test_configure_arm_joint_gains_light_damping_scoped_to_
    configured_sides_only` below covers that mechanism on its own.
    """
    import torch

    writes: dict[str, list] = {"stiffness": [], "damping": []}
    controller = object.__new__(DualArmController)
    controller.joint_groups = SimpleNamespace(
        left_arm=(0, 1, 2, 3, 4, 5, 6),
        right_arm=(7, 8, 9, 10, 11, 12, 13),
    )
    joint_pos = torch.zeros((1, 14))
    controller.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_stiffness=None,
            joint_damping=None,
            joint_pos=joint_pos,
        ),
        write_joint_stiffness_to_sim=lambda value, joint_ids: writes[
            "stiffness"
        ].append(tuple(joint_ids)),
        write_joint_damping_to_sim=lambda value, joint_ids: writes[
            "damping"
        ].append(tuple(joint_ids)),
    )

    original_light_sides = DualArmController._ARM_LIGHT_DAMPING_ONLY_SIDES
    DualArmController._ARM_LIGHT_DAMPING_ONLY_SIDES = ()
    try:
        controller._configure_arm_joint_gains(controller.robot)
    finally:
        DualArmController._ARM_LIGHT_DAMPING_ONLY_SIDES = original_light_sides

    touched_ids = {i for call in writes["stiffness"] for i in call}
    touched_ids |= {i for call in writes["damping"] for i in call}
    expected_ids = {
        i
        for side in DualArmController._ARM_GAIN_OVERRIDE_SIDES
        for i in getattr(controller.joint_groups, f"{side}_arm")
    }
    untouched_sides = {"left", "right"} - set(
        DualArmController._ARM_GAIN_OVERRIDE_SIDES
    )
    untouched_ids = {
        i
        for side in untouched_sides
        for i in getattr(controller.joint_groups, f"{side}_arm")
    }
    assert touched_ids == expected_ids
    assert touched_ids.isdisjoint(untouched_ids)


def test_configure_arm_joint_gains_light_damping_scoped_to_configured_sides_only():
    """The damping-only mechanism (`_ARM_LIGHT_DAMPING_ONLY_SIDES`, enabled
    2026-08-16 at an intermediate value per 8939f3a's own prescribed next
    step) must only touch the light wrist joints (5-7) of the sides it
    names, never the heavy joints (1-4) and never an unconfigured side --
    same shape of guarantee T2 above enforces for the other mechanism.
    """
    import torch

    writes: dict[str, list] = {"stiffness": [], "damping": []}
    controller = object.__new__(DualArmController)
    controller.joint_groups = SimpleNamespace(
        left_arm=(0, 1, 2, 3, 4, 5, 6),
        right_arm=(7, 8, 9, 10, 11, 12, 13),
    )
    joint_pos = torch.zeros((1, 14))
    controller.robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_stiffness=None,
            joint_damping=None,
            joint_pos=joint_pos,
        ),
        write_joint_stiffness_to_sim=lambda value, joint_ids: writes[
            "stiffness"
        ].append(tuple(joint_ids)),
        write_joint_damping_to_sim=lambda value, joint_ids: writes[
            "damping"
        ].append(tuple(joint_ids)),
    )

    original_override_sides = DualArmController._ARM_GAIN_OVERRIDE_SIDES
    DualArmController._ARM_GAIN_OVERRIDE_SIDES = ()
    try:
        controller._configure_arm_joint_gains(controller.robot)
    finally:
        DualArmController._ARM_GAIN_OVERRIDE_SIDES = original_override_sides

    touched_ids = {i for call in writes["damping"] for i in call}
    assert not writes["stiffness"], (
        "the damping-only mechanism must never write stiffness"
    )
    expected_ids = {
        i
        for side in DualArmController._ARM_LIGHT_DAMPING_ONLY_SIDES
        for i in getattr(controller.joint_groups, f"{side}_arm")[
            DualArmController._ARM_HEAVY_JOINT_COUNT :
        ]
    }
    heavy_ids = {
        i
        for side in ("left", "right")
        for i in getattr(controller.joint_groups, f"{side}_arm")[
            : DualArmController._ARM_HEAVY_JOINT_COUNT
        ]
    }
    assert touched_ids == expected_ids
    assert touched_ids.isdisjoint(heavy_ids)


def test_gripper_effort_scale_writes_scaled_then_authored_limit():
    calls = []
    controller = object.__new__(DualArmController)
    controller.robot = SimpleNamespace(
        write_joint_effort_limit_to_sim=lambda limit, joint_ids: calls.append(
            (limit, joint_ids)
        )
    )
    controller.joint_groups = SimpleNamespace(right_gripper=(7,))
    controller._default_gripper_effort_limits = {"right": 8.0}

    controller.set_gripper_effort_scale("right", 0.25)
    controller.restore_gripper_effort_limit("right")

    assert calls == [(2.0, (7,)), (8.0, (7,))]


@pytest.mark.parametrize("scale", (0.0, -0.1, 1.1, math.nan))
def test_gripper_effort_scale_rejects_invalid_values(scale):
    controller = object.__new__(DualArmController)
    controller.robot = SimpleNamespace(
        write_joint_effort_limit_to_sim=lambda *args, **kwargs: None
    )
    controller.joint_groups = SimpleNamespace(right_gripper=(7,))
    controller._default_gripper_effort_limits = {"right": 8.0}

    with pytest.raises(ValueError, match="effort scale"):
        controller.set_gripper_effort_scale("right", scale)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({}, True),
        ({"holding": False}, False),
        ({"held_ticks": 599}, False),
        ({"lifted_m": 0.0799}, False),
    ],
)
def test_grasp_lift_gate_uses_measured_object_outcome(overrides, expected):
    values = {
        "holding": True,
        "held_ticks": 600,
        "needed_ticks": 600,
        "lifted_m": 0.088,
        "min_lift_m": 0.08,
        **overrides,
    }
    assert grasp_lift_gate_passed(**values) is expected


def test_linear_ramp_target_clamps_at_end():
    assert linear_ramp_target(0.9, 0.0, 0, 4) == pytest.approx(0.9)
    assert linear_ramp_target(0.9, 0.0, 2, 4) == pytest.approx(0.45)
    assert linear_ramp_target(0.9, 0.0, 4, 4) == pytest.approx(0.0)
    assert linear_ramp_target(0.9, 0.0, 8, 4) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "args",
    [
        (math.nan, 0.0, 1, 2),
        (0.9, math.inf, 1, 2),
        (0.9, 0.0, -1, 2),
        (0.9, 0.0, 1, 0),
    ],
)
def test_linear_ramp_target_rejects_invalid_inputs(args):
    with pytest.raises(ValueError, match="ramp"):
        linear_ramp_target(*args)


@pytest.mark.parametrize("completed_steps", [0, 1, 3, 5, 8])
def test_synchronized_drag_targets_preserve_relative_offset(completed_steps):
    arm_start_y = -1.62
    anchor_start_y = -1.72
    starting_gap = arm_start_y - anchor_start_y
    arm_y, anchor_y = synchronized_drag_targets(
        arm_start_y, anchor_start_y, 0.26, completed_steps, 5
    )
    assert arm_y - anchor_y == pytest.approx(starting_gap)


def test_synchronized_drag_targets_ramp_endpoints():
    arm_y, anchor_y = synchronized_drag_targets(-1.62, -1.72, 0.26, 0, 5)
    assert arm_y == pytest.approx(-1.62)
    assert anchor_y == pytest.approx(-1.72)
    arm_y, anchor_y = synchronized_drag_targets(-1.62, -1.72, 0.26, 5, 5)
    assert arm_y == pytest.approx(-1.62 + 0.26)
    assert anchor_y == pytest.approx(-1.72 + 0.26)


def test_lift_ramps_vertical_target_before_accepting_convergence():
    controller = object.__new__(DualArmController)
    targets = []
    start = (1.0, 2.0, 0.8)
    quaternion = (1.0, 0.0, 0.0, 0.0)
    controller.ee_world_poses = lambda: (
        (start, quaternion),
        (start, quaternion),
    )
    controller.set_arm_target = lambda side, position, quat: targets.append(
        (side, position, quat)
    )
    controller.command = lambda: SimpleNamespace(
        left_succeeded=True, right_succeeded=True
    )
    controller.pose_error = lambda side, position, quat: (0.0, 0.0)
    controller._tracker = CartesianTargetTracker(
        TeleopTargets(
            left=Pose(start, quaternion),
            right=Pose(start, quaternion),
            left_gripper=0.0,
            right_gripper=0.0,
            spine=0.44,
        ),
        limits=TargetLimits(
            position_min=(-2.0, -2.0, -1.0),
            position_max=(2.0, 2.0, 3.0),
            spine_min=0.0,
            spine_max=0.85,
        ),
    )

    assert controller.lift(
        "right",
        0.3,
        step=lambda: None,
        dt=0.5,
        timeout_s=3.0,
        ramp_seconds=2.0,
        spine_assist_m=0.12,
    )
    assert all(target[1][:2] == (1.0, 2.0) for target in targets)
    assert [target[1][2] for target in targets] == pytest.approx(
        [0.875, 0.95, 1.025, 1.1]
    )
    assert controller.spine == pytest.approx(0.56)


def test_lift_calls_on_tick_every_tick_with_the_tick_index():
    # REV13 T4-followup-2 (plans/SYNC.md 2026-08-07): 9/9 episodes with a
    # telemetrically "sustained" close still failed to lift -- the close
    # loop was never the bottleneck. `on_tick` gives world_isaac.py the
    # same per-tick visibility into the lift motion that T2 gave the
    # close ramp, so a slip during lift is observable instead of inferred
    # from a single before/after rise measurement.
    controller = object.__new__(DualArmController)
    start = (1.0, 2.0, 0.8)
    quaternion = (1.0, 0.0, 0.0, 0.0)
    controller.ee_world_poses = lambda: (
        (start, quaternion),
        (start, quaternion),
    )
    controller.set_arm_target = lambda side, position, quat: None
    controller.command = lambda: SimpleNamespace(
        left_succeeded=True, right_succeeded=True
    )
    controller.pose_error = lambda side, position, quat: (0.0, 0.0)
    controller._tracker = CartesianTargetTracker(
        TeleopTargets(
            left=Pose(start, quaternion),
            right=Pose(start, quaternion),
            left_gripper=0.0,
            right_gripper=0.0,
            spine=0.44,
        ),
        limits=TargetLimits(
            position_min=(-2.0, -2.0, -1.0),
            position_max=(2.0, 2.0, 3.0),
            spine_min=0.0,
            spine_max=0.85,
        ),
    )

    seen_ticks = []
    assert controller.lift(
        "right",
        0.3,
        step=lambda: None,
        dt=0.5,
        timeout_s=3.0,
        ramp_seconds=2.0,
        spine_assist_m=0.0,
        on_tick=seen_ticks.append,
    )
    # timeout_s=3.0 / dt=0.5 -> 6 ticks total, indices 0..5.
    assert seen_ticks == [0, 1, 2, 3], (
        "loop returns as soon as convergence is accepted (tick 3, the "
        f"first tick at/after ramp_ticks=4), got {seen_ticks}"
    )


def test_lift_on_tick_defaults_to_none_and_is_optional():
    # Existing callers (and every fake in test_pipeline.py before this
    # change) never pass on_tick -- must stay a no-op, not a TypeError.
    controller = object.__new__(DualArmController)
    start = (1.0, 2.0, 0.8)
    quaternion = (1.0, 0.0, 0.0, 0.0)
    controller.ee_world_poses = lambda: (
        (start, quaternion),
        (start, quaternion),
    )
    controller.set_arm_target = lambda side, position, quat: None
    controller.command = lambda: SimpleNamespace(
        left_succeeded=True, right_succeeded=True
    )
    controller.pose_error = lambda side, position, quat: (0.0, 0.0)
    controller._tracker = CartesianTargetTracker(
        TeleopTargets(
            left=Pose(start, quaternion),
            right=Pose(start, quaternion),
            left_gripper=0.0,
            right_gripper=0.0,
            spine=0.44,
        ),
        limits=TargetLimits(
            position_min=(-2.0, -2.0, -1.0),
            position_max=(2.0, 2.0, 3.0),
            spine_min=0.0,
            spine_max=0.85,
        ),
    )

    assert controller.lift(
        "right", 0.3, step=lambda: None, dt=0.5, timeout_s=3.0,
        ramp_seconds=2.0, spine_assist_m=0.0,
    )


def _tracker(left: Pose, right: Pose) -> CartesianTargetTracker:
    return CartesianTargetTracker(
        TeleopTargets(
            left=left,
            right=right,
            left_gripper=0.04,
            right_gripper=0.04,
            spine=0.2,
        ),
        limits=TargetLimits(
            position_min=(-2.0, -2.0, -1.0),
            position_max=(2.0, 2.0, 3.0),
        ),
    )


def _assert_same_rotation(actual, expected, *, tolerance=1.0e-9):
    dot = abs(sum(a * b for a, b in zip(actual, expected)))
    assert dot == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("side", ("left", "right"))
def test_one_step_reach_command_lands_exactly_on_world_target(side):
    initial_left = Pose((-0.3, 0.4, 0.8), _quaternion_from_rpy(0.2, -0.1, 0.3))
    initial_right = Pose(
        (0.5, -0.2, 1.1), _quaternion_from_rpy(-0.4, 0.2, -0.5)
    )
    tracker = _tracker(initial_left, initial_right)
    before_other = (
        tracker.targets.right if side == "left" else tracker.targets.left
    )
    base_position = (-3.32, -1.72, 0.0)
    base_orientation = _quaternion_from_rpy(0.0, 0.0, math.pi)
    world_target = Pose(
        (-4.145, -1.75, 1.05),
        _quaternion_from_rpy(math.pi, 0.0, 0.0),
    )

    command = one_step_reach_command(
        getattr(tracker.targets, side),
        world_target,
        base_position,
        base_orientation,
        side=side,
        timestamp=12.5,
    )
    updated = tracker.apply(command)
    expected = pose_world_to_base(
        world_target, base_position, base_orientation
    )
    actual = getattr(updated, side)

    assert command.active
    assert command.source == "task3_autonomy.reach"
    assert command.timestamp == 12.5
    assert actual.position == pytest.approx(expected.position, abs=1.0e-9)
    _assert_same_rotation(actual.orientation_wxyz, expected.orientation_wxyz)
    assert (updated.right if side == "left" else updated.left) == before_other


def test_one_step_reach_command_rejects_unknown_side():
    pose = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="side"):
        one_step_reach_command(
            pose,
            pose,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            side="middle",
        )


# 2026-08-21: this table used to assert (0.9, True) and (1.049, True) --
# i.e. it encoded the bug as spec. Both sit PAST this asset's own measured
# mechanical open limit (0.8203 rad); a joint cannot be at 1.049 while
# holding anything, and 1.0145 was really observed on a gripper jammed
# fully open by contact overshoot and scored as a successful hold. The
# band now ends one contact-deflection BELOW the open limit, so "open" and
# "holding" are separable instead of 0.0003 rad apart.
@pytest.mark.parametrize(
    "position,expected",
    [
        (0.0, False),      # jaws met each other, nothing between them
        (0.049, False),
        (0.051, True),
        (0.4, True),
        (0.783, True),     # the one real measured grip (cup wall, 08-20)
        (0.79, False),     # inside the contact margin: effectively open
        (0.819, False),    # the false positive this fix exists for
        (0.8203, False),   # the authored open limit itself
        (1.049, False),    # past the mechanical limit; impossible
    ],
)
def test_gripper_holds_object(position, expected):
    assert gripper_holds_object(position) is expected


@pytest.mark.parametrize("width", (math.nan, math.inf, -math.inf))
def test_gripper_hold_predicate_rejects_nonfinite_width(width):
    with pytest.raises(ValueError, match="finite"):
        gripper_holds_object(width)


def test_gripper_hold_predicate_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="ordered"):
        gripper_holds_object(0.1, min_position_rad=0.3, max_position_rad=0.2)


class _FakeGripperJoint:
    """Plain-Python stand-in for a position-controlled joint under
    resistance -- no real robot needed. REV13 T2: `run_gripper_close_ramp`
    is a pure tick loop, so it should be exercisable with nothing more
    than a callable that tracks a commanded target with an optional
    floor (simulating contact resistance)."""

    def __init__(self, start: float, *, block_at: float | None = None):
        self.position = start
        self.block_at = block_at
        self.commanded: list[float] = []

    def set_target(self, target: float) -> None:
        self.commanded.append(target)
        if self.block_at is not None and target < self.block_at:
            self.position = self.block_at
        else:
            self.position = target

    def measured(self) -> float:
        return self.position


def test_run_gripper_close_ramp_closed_no_contact():
    """Nothing resists closure -- measured tracks commanded all the way
    to 0.0, matching the T7/T0 failure pattern (gripper_rad <= 0.13)."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9)
    telemetry: dict = {}
    holding = run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=10,
        ramp_ticks=5,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        telemetry=telemetry,
    )
    assert holding is False
    assert telemetry["tick_count"] == 10
    assert telemetry["contact_tick"] is None
    assert telemetry["outcome"] == "closed_no_contact"
    assert telemetry["final_residual_rad"] == pytest.approx(0.0, abs=1e-6)
    assert len(telemetry["ticks"]) == 10
    assert telemetry["ticks"][0]["tick"] == 1
    assert set(telemetry["ticks"][0]) == {
        "tick",
        "commanded_target_rad",
        "measured_position_rad",
        "error_rad",
    }


def test_run_gripper_close_ramp_contact_sustained():
    """An object blocks closure early and the block holds to the end --
    the proven-success shape (gripper_rad 0.2979, 0.6472)."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9, block_at=0.3)
    telemetry: dict = {}
    holding = run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=10,
        ramp_ticks=5,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        telemetry=telemetry,
    )
    assert holding is True
    assert telemetry["contact_tick"] is not None
    assert telemetry["outcome"] == "contact_sustained"
    assert telemetry["final_residual_rad"] == pytest.approx(0.3)


def test_run_gripper_close_ramp_contact_lost():
    """The object blocks closure briefly, then the block releases before
    the ramp ends -- a real momentary catch that did not survive the
    close, distinguishable for the first time from `closed_no_contact`."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9)
    telemetry: dict = {}

    def _set_target(target: float) -> None:
        joint.commanded.append(target)
        # Blocked (0.3) for the middle of the ramp, then releases and
        # tracks the commanded target again for the rest of the close.
        if 0.2 < target < 0.7:
            joint.position = 0.3
        else:
            joint.position = target

    holding = run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=10,
        ramp_ticks=5,
        set_target=_set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        telemetry=telemetry,
    )
    assert holding is False
    assert telemetry["contact_tick"] is not None
    assert telemetry["outcome"] == "contact_lost"
    assert telemetry["final_residual_rad"] == pytest.approx(0.0, abs=1e-6)


def test_run_gripper_close_ramp_rejects_jammed_fully_open():
    """Session 2026-08-15: the USD joint limit is 0..1 rad, but the old
    hardcoded `DEFAULT_HOLD_MAX_POSITION_RAD=1.05` let a gripper jammed past
    its own mechanical limit (measured 1.0145 during the first bimanual
    tray run) pass `min < position < max` and be scored as a successful
    hold. Passing the joint's real authored upper limit (here 1.0, standing
    in for `DualArmController._gripper_position_upper_limit`) must reject a
    block point past it, where the old default would have accepted it."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9, block_at=1.0145)
    telemetry: dict = {}
    holding = run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=10,
        ramp_ticks=5,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        hold_max_position_rad=1.0,
        telemetry=telemetry,
    )
    assert holding is False


def test_run_gripper_close_ramp_stall_lookback_filters_jitter():
    """2026-08-16 (cup): a real GPU close-ramp trace under sustained
    contact jittered a few thousandths of a radian tick to tick
    (outputs/task3_verify_grasp_lift/close_trace/close_ramp_ticks.json,
    e.g. 0.8905, 0.89042, 0.89742, 0.90636, 0.89852, 0.89159, 0.90286)
    even though the object was caught the whole time (error_rad grew
    steadily). Comparing only against the immediately-previous tick
    (`stall_lookback_ticks=1`, the default) reads every uptick as
    "closing" and the `stalled_ticks` counter never reaches
    `stall_ticks_required` -- this is reproduced here with a
    period-2 oscillation. Comparing against `stall_lookback_ticks` ago
    instead (same phase of a period-2 oscillation, so ~zero net change)
    must still freeze."""
    from task3_autonomy.arms import run_gripper_close_ramp

    tick = {"i": 0}

    def _measured() -> float:
        i = tick["i"]
        tick["i"] += 1
        if i < 25:
            return max(0.85, 0.9 - 0.002 * i)
        return 0.85 if i % 2 == 0 else 0.87

    telemetry_default: dict = {}
    run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=100,
        ramp_ticks=50,
        set_target=lambda target: None,
        measured_position=_measured,
        advance=lambda: None,
        contact_freeze_max_target_rad=0.95,
        hold_max_position_rad=1.0,
        telemetry=telemetry_default,
    )
    assert telemetry_default["held_target_rad"] is None

    tick["i"] = 0
    telemetry_lookback: dict = {}
    run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=100,
        ramp_ticks=50,
        set_target=lambda target: None,
        measured_position=_measured,
        advance=lambda: None,
        contact_freeze_max_target_rad=0.95,
        stall_lookback_ticks=10,
        hold_max_position_rad=1.0,
        telemetry=telemetry_lookback,
    )
    assert telemetry_lookback["held_target_rad"] is not None


def test_run_gripper_close_ramp_rejects_bad_stall_lookback_ticks():
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9)
    with pytest.raises(ValueError, match="stall_lookback_ticks"):
        run_gripper_close_ramp(
            start_position=0.9,
            end_position=0.0,
            total_ticks=10,
            ramp_ticks=5,
            set_target=joint.set_target,
            measured_position=joint.measured,
            advance=lambda: None,
            stall_lookback_ticks=0,
        )


def test_close_ramp_hold_on_contact_freezes_commanded_target():
    """REV13 T4: once contact is detected, the commanded target must stop
    moving -- the fix for T3's real GPU trace (contact detected tick 2,
    fingers shoved to 0.987 rad, held under resistance for ~190/300
    ticks, then ground down to closed-on-air in the final ~50 ticks
    because the ramp kept forcing the target toward 0 long after contact
    was already made)."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9, block_at=0.3)
    telemetry: dict = {}
    run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=10,
        ramp_ticks=5,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        hold_target_on_contact=True,
        telemetry=telemetry,
    )
    contact_tick = telemetry["contact_tick"]
    assert contact_tick is not None
    commanded = [t["commanded_target_rad"] for t in telemetry["ticks"]]
    assert commanded[0] != commanded[1], "the ramp must move before contact"
    frozen_value = commanded[contact_tick - 1]
    assert all(v == frozen_value for v in commanded[contact_tick - 1 :]), (
        f"commanded target must freeze at contact, got {commanded}"
    )
    assert telemetry["hold_target_on_contact"] is True
    assert telemetry["held_target_rad"] == pytest.approx(frozen_value)


def test_close_ramp_hold_on_contact_prevents_t3_failure_shape():
    """The exact geometry that produced `contact_lost` in the unfixed
    ramp (`test_run_gripper_close_ramp_contact_lost` above: blocked while
    the target is in (0.2, 0.7), released once the target is forced below
    0.2) must become `contact_sustained` once the target freezes at first
    contact and therefore never gets forced below the release point."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9)
    telemetry: dict = {}

    def _set_target(target: float) -> None:
        joint.commanded.append(target)
        if 0.2 < target < 0.7:
            joint.position = 0.3
        else:
            joint.position = target

    holding = run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=10,
        ramp_ticks=5,
        set_target=_set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        hold_target_on_contact=True,
        telemetry=telemetry,
    )
    assert holding is True
    assert telemetry["outcome"] == "contact_sustained"
    assert telemetry["final_residual_rad"] == pytest.approx(0.3)


def test_close_ramp_hold_on_contact_freezes_only_below_max_target():
    """REV13 T4-followup: T4's plain `hold_target_on_contact` shipped and
    FAILED on real GPU -- it froze at commanded target 0.923 rad
    (`proofs/2026-08-06_t4_bowl2_close_fix_reverted`), essentially still
    fully open, because contact was flagged the instant error crossed
    `contact_error_rad` regardless of how little the ramp had progressed.
    This reproduces that exact geometry (block_at=0.85, very close to the
    0.9 open start) and asserts the new `contact_freeze_max_target_rad`
    gate refuses to freeze there -- the ramp must keep commanding toward
    fully closed exactly as the unfixed default does."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9, block_at=0.85)
    telemetry: dict = {}
    holding = run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=100,
        ramp_ticks=90,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        hold_target_on_contact=True,
        contact_freeze_max_target_rad=0.8,
        telemetry=telemetry,
    )
    contact_tick = telemetry["contact_tick"]
    assert contact_tick is not None, (
        "the 0.85 floor must still register as contact"
    )
    target_at_contact = telemetry["ticks"][contact_tick - 1][
        "commanded_target_rad"
    ]
    assert target_at_contact > 0.8, (
        "test setup check: contact must be detected while still above the "
        f"freeze cutoff, got {target_at_contact}"
    )
    assert telemetry["held_target_rad"] is None, (
        "must NOT freeze -- contact was detected too early in the ramp "
        "to represent a usable grip (T4's real failure shape)"
    )
    commanded = [t["commanded_target_rad"] for t in telemetry["ticks"]]
    assert commanded[-1] == pytest.approx(0.0, abs=1e-6), (
        "without a freeze, the ramp must still reach fully closed by the end"
    )
    # Not asserting on `holding` here: this fake joint's `block_at` is a
    # permanent floor (it never lets go on its own), which is a
    # simplification real hardware does not share -- T4's real GPU trace
    # shows the object CAN keep slipping further even without a freeze
    # (measured position climbed past the frozen target in T4's own data).
    # The behavior this test exists to prove is the freeze DECISION
    # (`held_target_rad is None` above), not this fake's downstream
    # `holding` value.
    del holding


def test_close_ramp_hold_on_contact_freezes_below_max_target():
    """The mirror case: T3's real trace froze `hold_target_on_contact`
    should have applied at commanded target 0.647 rad -- comfortably
    below the 0.8 cutoff, and almost exactly this project's own proven
    upper hold value (0.6472). This reproduces that shape (block_at=0.6)
    and asserts the freeze DOES fire and lands in a plausible band."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9, block_at=0.6)
    telemetry: dict = {}
    holding = run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=100,
        ramp_ticks=90,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        hold_target_on_contact=True,
        contact_freeze_max_target_rad=0.8,
        telemetry=telemetry,
    )
    contact_tick = telemetry["contact_tick"]
    assert contact_tick is not None
    target_at_contact = telemetry["ticks"][contact_tick - 1][
        "commanded_target_rad"
    ]
    assert target_at_contact <= 0.8, (
        "test setup check: contact must be detected at/below the freeze "
        f"cutoff, got {target_at_contact}"
    )
    assert telemetry["held_target_rad"] == pytest.approx(target_at_contact)
    assert holding is True
    assert telemetry["outcome"] == "contact_sustained"
    commanded = [t["commanded_target_rad"] for t in telemetry["ticks"]]
    assert commanded[-1] == pytest.approx(target_at_contact), (
        "the frozen target must not keep moving after the freeze"
    )


def test_run_gripper_close_ramp_rejects_nonpositive_ticks():
    from task3_autonomy.arms import run_gripper_close_ramp

    with pytest.raises(ValueError, match="positive"):
        run_gripper_close_ramp(
            start_position=0.9,
            end_position=0.0,
            total_ticks=0,
            ramp_ticks=5,
            set_target=lambda target: None,
            measured_position=lambda: 0.0,
            advance=lambda: None,
        )


def test_ordered_joint_targets_converts_lula_mappingproxy_to_sequence():
    targets = MappingProxyType({"joint_b": 2.0, "joint_a": 1.0})
    assert ordered_joint_targets(targets, ("joint_a", "joint_b")) == [1.0, 2.0]
    assert ordered_joint_targets(MappingProxyType({}), ("joint_a",)) is None


def test_ordered_joint_targets_rejects_missing_joint():
    with pytest.raises(ValueError, match="missing joint joint_b"):
        ordered_joint_targets({"joint_a": 1.0}, ("joint_a", "joint_b"))


def test_close_ramp_freezes_where_the_object_stops_the_jaws():
    """The jaws must STOP where the object stopped them, not crush past it.

    Measured, stiff_1: with gripper stiffness raised to 60 the close finally
    had authority, stopped on the spoon at 0.4077 rad -- and then kept going
    to 0.0011, ejecting it before the carry ever started. Detecting the
    stall (measured stops following a still-closing command) is a stronger
    signal than error-crosses-threshold, which fired at tick 4-9 in every
    close this project has ever logged regardless of what was there.
    """
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9, block_at=0.40)
    telemetry: dict = {}
    run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=300,
        ramp_ticks=200,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        telemetry=telemetry,
    )
    final_cmd = telemetry["ticks"][-1]["commanded_target_rad"]
    assert final_cmd > 0.2, (
        "the ramp should have frozen near the object's width, not driven to "
        f"fully closed; final command was {final_cmd}"
    )


def test_close_ramp_stall_freeze_still_refuses_a_near_open_stall():
    """A stall at 0.85 is the T4 failure, not an object: freezing there
    grips nothing. The progress gate must still refuse it."""
    from task3_autonomy.arms import run_gripper_close_ramp

    joint = _FakeGripperJoint(0.9, block_at=0.85)
    telemetry: dict = {}
    run_gripper_close_ramp(
        start_position=0.9,
        end_position=0.0,
        total_ticks=200,
        ramp_ticks=150,
        set_target=joint.set_target,
        measured_position=joint.measured,
        advance=lambda: None,
        telemetry=telemetry,
    )
    final_cmd = telemetry["ticks"][-1]["commanded_target_rad"]
    assert final_cmd < 0.1, (
        "a near-open stall must not freeze the ramp; final command was "
        f"{final_cmd}"
    )


# --------------------------------------------------------------------------- #
# 2026-08-21 (DOCTOR.md 4.1): the joint-thrash guard used to run AFTER
# command() had written the flipped target and step() had executed it, so it
# reported an accident it was positioned too late to prevent. Measured on GPU
# (pinbase_yaw_fix_run3, ticks 6173-6280): joint_thrash_bailed fired on the
# same tick the cup was flung off the table onto the floor. reach() now solves,
# inspects, and only then commits.
# --------------------------------------------------------------------------- #


class _Column:
    """Minimal stand-in for `tensor[0, ids].tolist()` indexing."""

    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, key):
        _row, ids = key
        return _Column([self._values[i] for i in ids])

    def tolist(self):
        return list(self._values)


class _IKResult:
    left_succeeded = True
    right_succeeded = True


class _RecordingArms(DualArmController):
    """A DualArmController whose IK flips solutions on a chosen tick.

    Built with `object.__new__` so no Isaac import is needed: `reach()` only
    touches the handful of attributes set below.
    """

    def __init__(self, flip_on_tick, flip_to):
        self._tick = 0
        self._flip_on_tick = flip_on_tick
        self._flip_to = flip_to
        self._safe = [0.10] * 7
        self._position_targets = _Column(self._safe)
        self.committed = []
        self.steps = 0

        class _Groups:
            left_arm = tuple(range(7))
            right_arm = tuple(range(7))

        class _Data:
            joint_pos = _Column([0.10] * 7)

        class _Robot:
            data = _Data()

        self.joint_groups = _Groups()
        self.robot = _Robot()

    def set_arm_target(self, side, position, quat_wxyz):
        return None

    def solve_position_targets(self):
        values = (
            list(self._flip_to)
            if self._tick == self._flip_on_tick
            else list(self._safe)
        )
        self._tick += 1
        return _IKResult(), _Column(values)

    def commit_position_targets(self, composed):
        self.committed.append(composed.tolist())
        self._position_targets = composed

    def pose_error(self, side, position, quat_wxyz):
        return 0.5, 0.0  # never converges, so the loop runs its full budget


def _step_counter(arms):
    def _step():
        arms.steps += 1

    return _step


def test_reach_never_commits_a_joint_flip_it_bails_on():
    """The flipped solution must never reach the robot.

    This is the whole point of the fix: a guard that fires after
    `set_joint_position_target` has already been called is telemetry. The
    articulation must be left holding its last SAFE target instead.
    """
    flip = [3.7] + [0.10] * 6  # 3.6 rad on joint 1, the real measured value
    arms = _RecordingArms(flip_on_tick=3, flip_to=flip)

    ok = arms.reach(
        "left",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        step=_step_counter(arms),
        dt=0.005,
        timeout_s=0.05,  # 10 ticks of budget
        max_joint_delta_rad=0.5,
    )

    assert ok is False
    assert arms.committed, "expected the safe ticks before the flip to commit"
    for committed in arms.committed:
        assert committed[0] != 3.7, (
            "the flipped solution was written to the robot -- this is the "
            "post-hoc-guard bug the split was made to fix"
        )
    # Three safe ticks committed and stepped, then the flip is rejected
    # without a commit and without a step.
    assert len(arms.committed) == 3
    assert arms.steps == 3


def test_reach_still_commits_normally_when_no_flip_occurs():
    """The reordering must not change the un-flipped path."""
    arms = _RecordingArms(flip_on_tick=None, flip_to=None)

    arms.reach(
        "left",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        step=_step_counter(arms),
        dt=0.005,
        timeout_s=0.05,
        max_joint_delta_rad=0.5,
    )

    assert len(arms.committed) == 10
    assert arms.steps == 10


# --------------------------------------------------------------------------- #
# 2026-08-21 (DOCTOR.md 4.3): the hold band's upper bound WAS the jaws' open
# limit, so "holding something" and "open, holding nothing" were 0.0003 rad
# apart. Both gates -- the band and honest_hold's own `< GRIPPER_OPEN_RAD` --
# admitted a gripper at 0.819.
# --------------------------------------------------------------------------- #

MEASURED_OPEN_RAD = 0.8203047513961792  # authored USD limit, both sides


def test_a_fully_open_gripper_does_not_count_as_holding():
    """The exact false positive: open, closed on air, scored as a grasp."""
    assert not gripper_holds_object(0.819, max_position_rad=MEASURED_OPEN_RAD)
    assert not gripper_holds_object(
        MEASURED_OPEN_RAD, max_position_rad=MEASURED_OPEN_RAD
    )


def test_the_one_real_measured_grip_still_counts_as_holding():
    """The margin must not be so wide it rejects a genuine grasp.

    2026-08-20 gamepad session: the gripper was commanded fully closed and
    the real joint reading stopped at 0.783 rad -- owner-confirmed as the
    cup's own wall wedged between the pads. That is the only physically
    verified loaded grip this project has, so it is the binding constraint
    on how large the margin may be.
    """
    assert gripper_holds_object(0.783, max_position_rad=MEASURED_OPEN_RAD)


def test_hold_band_upper_bound_is_open_limit_minus_contact_margin():
    """The margin is derived from this codebase's own contact definition."""
    upper = MEASURED_OPEN_RAD - DEFAULT_CONTACT_ERROR_RAD
    assert gripper_holds_object(
        upper - 1e-6, max_position_rad=MEASURED_OPEN_RAD
    )
    assert not gripper_holds_object(
        upper + 1e-6, max_position_rad=MEASURED_OPEN_RAD
    )


def test_closed_on_nothing_still_does_not_count_as_holding():
    """The closed end is unchanged -- jaws met each other, not an object."""
    assert not gripper_holds_object(0.0, max_position_rad=MEASURED_OPEN_RAD)
    assert not gripper_holds_object(0.01, max_position_rad=MEASURED_OPEN_RAD)


def test_hold_margin_must_leave_an_ordered_band():
    with pytest.raises(ValueError):
        gripper_holds_object(0.4, max_position_rad=0.06, hold_margin_rad=0.03)
    with pytest.raises(ValueError):
        gripper_holds_object(0.4, hold_margin_rad=-0.01)


# --------------------------------------------------------------------------- #
# 2026-08-21 (DOCTOR.md 4.4): sync_targets_from_measured() rebuilt the tracker
# with the MEASURED gripper position, so any code path that re-anchored while
# holding threw the grip command away. navigate_to() does exactly that on the
# carry path.
# --------------------------------------------------------------------------- #


class _SyncArms(DualArmController):
    """DualArmController with only what sync_targets_from_measured touches."""

    def __init__(self, commanded_rad, measured_rad):
        from teleop_targets import (
            CartesianTargetTracker,
            Pose,
            TargetLimits,
            TeleopTargets,
        )

        self._CartesianTargetTracker = CartesianTargetTracker
        self._TargetLimits = TargetLimits
        self._pose = Pose
        self._targets_cls = TeleopTargets
        self.measured_rad = measured_rad

        class _Groups:
            left_gripper = (0,)
            right_gripper = (1,)

        class _Scalar:
            def __init__(self, v):
                self._v = v

            def item(self):
                return self._v

        class _Data:
            joint_pos = [[_Scalar(measured_rad), _Scalar(measured_rad)]]
            joint_pos_limits = [[[0.0, 0.82], [0.0, 0.82]]]

        class _Robot:
            data = _Data()

        self.joint_groups = _Groups()
        self.robot = _Robot()

        rest = Pose((0.4, 0.0, 0.9), (1.0, 0.0, 0.0, 0.0))
        self._tracker = CartesianTargetTracker(
            TeleopTargets(
                left=rest,
                right=rest,
                left_gripper=commanded_rad,
                right_gripper=commanded_rad,
                spine=0.4,
            ),
            limits=TargetLimits(
                position_min=(-1.5, -1.5, -0.5),
                position_max=(1.5, 1.5, 2.5),
                gripper_min=0.0,
                gripper_max=0.82,
                spine_min=0.0,
                spine_max=0.85,
            ),
        )

    # Everything below stands in for the live-robot reads.
    def _root_pose(self, robot):
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)

    def _measured_spine(self):
        return 0.4

    def _measured_position_targets(self, robot):
        return None

    def _gripper_position_lower_limit(self, side):
        return 0.0

    def _gripper_position_upper_limit(self, side):
        return 0.82

    class _IK:
        @staticmethod
        def current_end_effector_poses(pos, quat, spine):
            return (
                ((0.4, 0.2, 0.9), (1.0, 0.0, 0.0, 0.0)),
                ((0.4, -0.2, 0.9), (1.0, 0.0, 0.0, 0.0)),
            )

    _ik = _IK()


def test_sync_preserves_the_commanded_grip_by_default():
    """A loaded grip survives a re-anchor.

    Commanded fully closed, measured 0.783 (the cup's own wall holding the
    jaws apart -- the real 2026-08-20 signature). Re-anchoring to measured
    would replace "squeeze" with "sit where the object put you".
    """
    arms = _SyncArms(commanded_rad=0.0, measured_rad=0.783)
    arms.sync_targets_from_measured()
    assert arms._tracker.targets.left_gripper == 0.0
    assert arms._tracker.targets.right_gripper == 0.0


def test_sync_can_still_re_anchor_the_gripper_when_asked():
    arms = _SyncArms(commanded_rad=0.0, measured_rad=0.783)
    arms.sync_targets_from_measured(preserve_gripper=False)
    assert arms._tracker.targets.left_gripper == pytest.approx(0.783)
