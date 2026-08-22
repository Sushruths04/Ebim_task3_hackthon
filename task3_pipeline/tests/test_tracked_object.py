# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrackedObject and the typed perception-recovery table."""

from __future__ import annotations

import math
import time

import pytest

from task3_pipeline.outcomes import SkillOutcome
from task3_pipeline.recovery import (
    RecoveryAction,
    decide,
    perception_outcome,
)
from task3_pipeline.tracked_object import TrackedObject, TrackStatus


def make(**kw) -> TrackedObject:
    base = dict(
        object_id="cup#1",
        label="cup",
        position=(1.0, 2.0, 0.75),
        stamp_s=10.0,
        confidence=0.9,
        status=TrackStatus.CONFIRMED,
    )
    base.update(kw)
    return TrackedObject(**base)


# --- freshness ---------------------------------------------------------- #


def test_fresh_track_is_not_stale():
    obj = make(stamp_s=10.0)
    assert obj.age_s(10.5) == pytest.approx(0.5)
    assert not obj.is_stale(10.5, max_age_s=1.0)


def test_old_track_is_stale():
    obj = make(stamp_s=10.0)
    assert obj.is_stale(12.0, max_age_s=1.0)


def test_future_stamp_counts_as_stale_not_fresh():
    """A negative age means the clocks disagree; that must not read as fresh."""
    obj = make(stamp_s=20.0)
    assert obj.age_s(15.0) < 0.0
    assert obj.is_stale(15.0, max_age_s=100.0)


def test_wall_clock_timestamp_is_rejected_loudly():
    """The failure this guard exists for: mixing time.time() with sim time."""
    with pytest.raises(ValueError, match="wall-clock epoch"):
        make(stamp_s=time.time()).is_stale(12.0, max_age_s=1.0)

    obj = make(stamp_s=10.0)
    with pytest.raises(ValueError, match="wall-clock epoch"):
        obj.is_stale(time.time(), max_age_s=1.0)


def test_negative_max_age_is_an_error():
    with pytest.raises(ValueError):
        make().is_stale(11.0, max_age_s=-1.0)


# --- confidence --------------------------------------------------------- #


def test_confident_when_above_tau_and_confirmed():
    assert make(confidence=0.9, status=TrackStatus.CONFIRMED).is_confident(0.8)


def test_not_confident_below_tau():
    assert not make(confidence=0.5).is_confident(0.8)


@pytest.mark.parametrize("status", [TrackStatus.TENTATIVE, TrackStatus.LOST])
def test_tentative_and_lost_are_never_confident(status):
    """High score on an unactable track must not read as 'confident'."""
    assert not make(confidence=0.99, status=status).is_confident(0.8)


def test_coasting_can_still_be_confident():
    """Coasting means 'not currently visible', not 'unknown'."""
    assert make(confidence=0.9, status=TrackStatus.COASTING).is_confident(0.8)


def test_tau_out_of_range_is_an_error():
    with pytest.raises(ValueError):
        make().is_confident(1.5)


def test_is_actionable_requires_both():
    obj = make(stamp_s=10.0, confidence=0.9)
    assert obj.is_actionable(10.2, max_age_s=1.0, tau=0.8)
    assert not obj.is_actionable(99.0, max_age_s=1.0, tau=0.8)  # stale
    assert not make(confidence=0.1).is_actionable(10.2, max_age_s=1.0, tau=0.8)


# --- geometry ----------------------------------------------------------- #


def test_covariance_none_is_distinct_from_zero():
    assert make().position_sigma_m() is None
    assert make(position_covariance=(0.0, 0.0, 0.0)).position_sigma_m() == 0.0


def test_position_sigma_is_root_sum():
    obj = make(position_covariance=(0.01, 0.01, 0.02))
    assert obj.position_sigma_m() == pytest.approx(math.sqrt(0.04))


def test_distance_to():
    assert make(position=(0.0, 0.0, 0.0)).distance_to((3.0, 4.0, 0.0)) == 5.0


# --- recovery table ------------------------------------------------------ #


def test_perception_outcome_prefers_stale_over_low_confidence():
    assert (
        perception_outcome(is_stale=True, is_confident=False)
        is SkillOutcome.PERCEPTION_STALE
    )


def test_perception_outcome_low_confidence():
    assert (
        perception_outcome(is_stale=False, is_confident=False)
        is SkillOutcome.PERCEPTION_LOW_CONFIDENCE
    )


def test_perception_outcome_none_when_actionable():
    assert perception_outcome(is_stale=False, is_confident=True) is None


@pytest.mark.parametrize(
    "outcome",
    [SkillOutcome.PERCEPTION_STALE, SkillOutcome.PERCEPTION_LOW_CONFIDENCE],
)
def test_perception_failures_reperceive_and_are_free(outcome):
    """REV20 P4.5: re-perceive instead of acting, without spending budget."""
    d = decide(outcome)
    assert d.action is RecoveryAction.REPERCEIVE
    assert d.free is True


def test_reperception_is_bounded():
    """An absent object must fail, not loop forever looking like a hang."""
    d = decide(SkillOutcome.PERCEPTION_STALE, reperceive_attempts=2, max_reperceive=2)
    assert d.action is RecoveryAction.ABORT_OBJECT


def test_timeout_abandons_rather_than_respending_the_budget():
    assert decide(SkillOutcome.TIMEOUT).action is RecoveryAction.ABORT_OBJECT


def test_nav_short_goes_back_to_the_navigator():
    assert decide(SkillOutcome.NAV_SHORT).action is RecoveryAction.REPLAN_NAV


def test_success_proceeds():
    assert decide(SkillOutcome.SUCCESS).action is RecoveryAction.PROCEED


@pytest.mark.parametrize(
    "outcome",
    [
        SkillOutcome.IK_FAIL,
        SkillOutcome.WEAK_GRASP,
        SkillOutcome.MISS,
        SkillOutcome.SLIP,
    ],
)
def test_existing_outcomes_still_retry_with_param(outcome):
    """Regression guard: the outcomes RetryPolicy already handled are unchanged."""
    d = decide(outcome)
    assert d.action is RecoveryAction.RETRY_WITH_PARAM
    assert d.free is False
