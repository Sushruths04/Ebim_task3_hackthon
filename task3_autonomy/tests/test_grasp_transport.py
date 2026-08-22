# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the shared grip-quality/verified_* contract (sec 16.5,
sec 4.38 amendment item 7)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from task3_autonomy.grasp_transport import (  # noqa: E402
    DEFAULT_HOLD_MAX_DISTANCE_M,
    OBJECT_INTEGRITY_TOL_M,
    classify_grip_quality,
    close_failure_reason,
    contact_force_state,
    hold_failure_reason,
    independent_signals,
    object_disturbed,
    object_follows_end_effector,
    verified_close,
    verified_hold,
    verified_lift,
    verified_placement,
)

# ---- P0.5: object-keyed grasp bands (handoff sec 17.4 #3) --------------- #


def test_classify_grip_quality_default_object_matches_cup_band() -> None:
    # No object_name passed -- every pre-existing caller in the codebase and
    # every test above this block -- must be bit-for-bit the cup band.
    assert classify_grip_quality(0.0709) == "caged"
    assert classify_grip_quality(0.0521) == "marginal"
    assert classify_grip_quality(1.0091) == "open"


def test_classify_grip_quality_cup_explicit_matches_default() -> None:
    for rad in (0.0, 0.0433, 0.065, 0.076, 0.09, 0.0901, 0.91):
        assert classify_grip_quality(rad, "cup") == classify_grip_quality(rad)


def test_classify_grip_quality_bowl2_uses_its_own_wider_band() -> None:
    # sec 4.49: bowl2 closed to 0.205 rad -- "a real, if loose... partial
    # closure", misclassified "open" by the cup band. Its own band must
    # accept it.
    assert classify_grip_quality(0.205, "bowl2") == "caged"
    # And the cup band must still reject the same angle for cup.
    assert classify_grip_quality(0.205, "cup") == "open"


def test_classify_grip_quality_contact_objects_use_force_not_angle() -> None:
    # sec 4.49: spoon2 closed to -0.0 rad (looks like "air" by angle) and
    # plate2 to 1.0267 rad (looks "open" by angle) -- both on nothing. A
    # real grip on either object must be classified "caged" purely from
    # contact force, regardless of what the angle says.
    assert (
        classify_grip_quality(-0.0, "spoon2", contact_force_n=0.4) == "caged"
    )
    assert (
        classify_grip_quality(1.0267, "plate2", contact_force_n=0.4) == "caged"
    )
    assert classify_grip_quality(-0.0, "spoon2", contact_force_n=0.0) == "air"
    assert classify_grip_quality(-0.0, "spoon2") == "unknown"


def test_close_failure_reason_handles_unknown() -> None:
    assert (
        close_failure_reason("unknown")
        == "close_unclassifiable_no_contact_sensor"
    )


def test_verified_hold_default_object_matches_cup_band() -> None:
    # Same >=2-signal call, only the new object_name kwarg differs from the
    # pre-existing tests above -- confirms the default is unchanged.
    assert (
        verified_hold(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=0.0709,
            object_name="cup",
        )
        is True
    )
    assert (
        verified_hold(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=0.205,
            object_name="cup",
        )
        is False
    )


def test_verified_hold_bowl2_band_accepts_its_own_angle() -> None:
    assert (
        verified_hold(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=0.205,
            object_name="bowl2",
        )
        is True
    )


def test_classify_grip_quality_air() -> None:
    assert classify_grip_quality(0.0) == "air"
    assert classify_grip_quality(0.0007) == "air"
    assert classify_grip_quality(0.0081) == "air"


def test_classify_grip_quality_marginal_band() -> None:
    assert classify_grip_quality(0.02) == "marginal"
    assert classify_grip_quality(0.0433) == "marginal"
    assert classify_grip_quality(0.0521) == "marginal"
    # Just below the caged lower boundary is still marginal, not caged.
    assert classify_grip_quality(0.0649) == "marginal"


def test_classify_grip_quality_caged_lower_boundary() -> None:
    # sec 15.3: the band is inclusive at 0.065.
    assert classify_grip_quality(0.065) == "caged"


def test_classify_grip_quality_caged_valid_values() -> None:
    assert classify_grip_quality(0.0709) == "caged"  # frozen proof
    assert classify_grip_quality(0.076) == "caged"  # run18
    assert classify_grip_quality(0.0855) == "caged"  # observed reproduction


def test_classify_grip_quality_caged_upper_boundary() -> None:
    # sec 15.3: the band is inclusive at 0.09; just above it is "open".
    assert classify_grip_quality(0.09) == "caged"
    assert classify_grip_quality(0.0901) == "open"


def test_classify_grip_quality_wide_open() -> None:
    # ep5's 1.0091 wide-open case (sec 4.29 Finding 2): must NOT read as
    # caged even though the gripper genuinely closed by some amount.
    assert classify_grip_quality(0.91) == "open"
    assert classify_grip_quality(0.9885) == "open"
    assert classify_grip_quality(1.0091) == "open"


def test_verified_close_requires_caged_band() -> None:
    assert verified_close(classify_grip_quality(0.0709)) is True
    assert verified_close(classify_grip_quality(0.0521)) is False
    assert verified_close(classify_grip_quality(1.0091)) is False


def test_close_failure_reason_matches_verified_close() -> None:
    assert close_failure_reason(classify_grip_quality(0.0709)) is None
    assert (
        close_failure_reason(classify_grip_quality(0.0))
        == "close_air_no_contact"
    )
    assert (
        close_failure_reason(classify_grip_quality(0.0521))
        == "close_marginal_below_caged_band"
    )
    assert (
        close_failure_reason(classify_grip_quality(1.0091))
        == "close_wide_open_never_closed"
    )


def test_object_follows_end_effector_distance_boundary() -> None:
    below = (0.0, 0.0, 0.0), (0.0, 0.0, 0.14)
    at = (0.0, 0.0, 0.0), (0.0, 0.0, DEFAULT_HOLD_MAX_DISTANCE_M)
    above = (0.0, 0.0, 0.0), (0.0, 0.0, 0.16)
    assert object_follows_end_effector(*below) is True
    assert object_follows_end_effector(*at) is True
    assert object_follows_end_effector(*above) is False


def test_verified_lift() -> None:
    assert verified_lift(True) is True
    assert verified_lift(False) is False


def test_verified_hold_requires_two_signals_ok() -> None:
    # Caged angle + object still rising + still tracking the EE: verified.
    assert (
        verified_hold(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=0.0709,
        )
        is True
    )
    # Contact force substitutes for a caged angle when the sensor is wired.
    assert (
        verified_hold(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=0.4,
            gripper_position_rad=0.02,
        )
        is True
    )


def test_verified_hold_rejects_ep5_style_false_positive() -> None:
    # arms.py:29's DEFAULT_HOLD_MIN_POSITION_RAD >= 0.05 one-sided test read
    # ok:true at 1.0091 rad (sec 4.37/16.2 item 2) -- the wide-open gripper
    # case. verified_hold must reject this even if object_rose/follows are
    # both true, since 1.0091 rad is neither caged nor force-confirmed.
    assert (
        verified_hold(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=1.0091,
        )
        is False
    )


def test_verified_hold_rejects_missing_rise_or_tracking() -> None:
    assert (
        verified_hold(
            object_rose=False,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=0.0709,
        )
        is False
    )
    assert (
        verified_hold(
            object_rose=True,
            object_follows_ee=False,
            grasp_force_n=None,
            gripper_position_rad=0.0709,
        )
        is False
    )


def test_hold_failure_reason_names_first_failing_signal() -> None:
    assert (
        hold_failure_reason(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=0.0709,
        )
        is None
    )
    assert (
        hold_failure_reason(
            object_rose=False,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=0.0709,
        )
        == "hold_object_did_not_rise"
    )
    assert (
        hold_failure_reason(
            object_rose=True,
            object_follows_ee=False,
            grasp_force_n=None,
            gripper_position_rad=0.0709,
        )
        == "hold_object_separated_from_ee"
    )
    assert (
        hold_failure_reason(
            object_rose=True,
            object_follows_ee=True,
            grasp_force_n=None,
            gripper_position_rad=1.0091,
        )
        == "hold_lost_grip_signal"
    )


def test_independent_signals_none_is_not_agreement() -> None:
    signals = independent_signals(
        gripper_position_rad=0.0709,
        contact_force_n=None,
        object_rose=None,
        object_follows_ee=None,
    )
    assert signals == ["gripper_rad"]


def test_contact_force_state_distinguishes_missing_sensor_from_zero() -> None:
    # sec 4.38 amendment item 7: sec 4.37 already conflated "no sensor" with
    # "sensor read exactly 0.0N" once -- these must never collapse together.
    assert contact_force_state(None) == "unavailable"
    assert contact_force_state(0.0) == "below_threshold"
    assert contact_force_state(0.005) == "below_threshold"
    assert contact_force_state(0.4) == "detected"
    # independent_signals treats both "unavailable" and "below_threshold"
    # identically for the boolean gate (neither counts as agreement) --
    # confirm that identical gate outcome without erasing the state
    # distinction contact_force_state still reports separately.
    no_sensor = independent_signals(contact_force_n=None)
    zero_reading = independent_signals(contact_force_n=0.0)
    assert no_sensor == zero_reading == []
    assert contact_force_state(None) != contact_force_state(0.0)


def test_verified_placement_mirrors_the_real_scorer() -> None:
    assert verified_placement(True) is True
    assert verified_placement(False) is False


def test_object_disturbed_within_tolerance_is_false() -> None:
    start = (-4.185, -1.753, 0.747)
    live = (-4.185 + 0.01, -1.753, 0.747)  # 1cm, below the 3cm default
    assert object_disturbed(start, live) is False


def test_object_disturbed_at_boundary_is_false() -> None:
    start = (0.0, 0.0, 0.0)
    live = (OBJECT_INTEGRITY_TOL_M, 0.0, 0.0)
    assert object_disturbed(start, live) is False


def test_object_disturbed_beyond_tolerance_is_true() -> None:
    # sec 17.4 #1's Stage-1 cup case: 0.31m of world-frame motion.
    start = (-4.185, -1.753, 0.747)
    live = (-4.185, -1.753 - 0.31, 0.747)
    assert object_disturbed(start, live) is True


def test_object_disturbed_custom_tolerance() -> None:
    start = (0.0, 0.0, 0.0)
    live = (0.05, 0.0, 0.0)
    assert object_disturbed(start, live, tol_m=0.03) is True
    assert object_disturbed(start, live, tol_m=0.1) is False
