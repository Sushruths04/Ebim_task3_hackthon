# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REV12 T3: safety-vocabulary translation + fully_completed reporting.

Before this fix, ``orchestrator._safety_flags`` read metrics keys
("collision"/"watchdog"/"dropped") that no producer in this codebase ever
sets -- the fail-closed chain's safety gate was silently dead on every
real run (see ``task3_pipeline/safety.py``'s module docstring). These
tests prove each real producer signal now reaches the chain's terminal
predicate, one test per translation, plus that a stage can never report
completion off a caught exception/timeout path.

Pure CPU -- no Isaac, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from task3_pipeline import config  # noqa: E402
from task3_pipeline.orchestrator import _safety_flags  # noqa: E402
from task3_pipeline.outcomes import SkillOutcome, SkillReport  # noqa: E402
from task3_pipeline.safety import (  # noqa: E402
    SafetyEvent,
    translate_report,
)
from task3_pipeline.stages import StageResult  # noqa: E402


def _report(skill="navigate", outcome=SkillOutcome.SUCCESS, metrics=None):
    return SkillReport(skill=skill, outcome=outcome, metrics=metrics or {})


# --------------------------------------------------------------------------- #
# One test per translation-table entry.
# --------------------------------------------------------------------------- #


def test_stalled_metric_translates_to_watchdog_stall():
    report = _report(metrics={"stalled": True})
    assert translate_report(report) == {SafetyEvent.WATCHDOG_STALL}


def test_no_stall_produces_no_watchdog_event():
    report = _report(metrics={"stalled": False})
    assert SafetyEvent.WATCHDOG_STALL not in translate_report(report)


def test_slip_outcome_translates_to_object_dropped():
    report = _report(skill="hold", outcome=SkillOutcome.SLIP, metrics={})
    assert translate_report(report) == {SafetyEvent.OBJECT_DROPPED}


def test_non_slip_outcome_produces_no_dropped_event():
    report = _report(skill="hold", outcome=SkillOutcome.SUCCESS, metrics={})
    assert SafetyEvent.OBJECT_DROPPED not in translate_report(report)


def test_head_force_over_threshold_translates_to_collision():
    report = _report(
        skill="feed_hold",
        metrics={"peak_head_force_n": config.HEAD_MAX_FORCE_N + 1.0},
    )
    assert translate_report(report) == {
        SafetyEvent.HEAD_CONTACT_FORCE_EXCEEDED
    }


def test_head_force_under_threshold_produces_no_collision_event():
    report = _report(
        skill="feed_hold",
        metrics={"peak_head_force_n": config.HEAD_MAX_FORCE_N - 1.0},
    )
    assert SafetyEvent.HEAD_CONTACT_FORCE_EXCEEDED not in translate_report(
        report
    )


def test_head_force_unmeasured_produces_no_collision_event():
    """Absence means 'not measured here', never a false collision-free
    claim."""
    report = _report(skill="reach", metrics={})
    assert SafetyEvent.HEAD_CONTACT_FORCE_EXCEEDED not in translate_report(
        report
    )


def test_clean_report_produces_no_events():
    report = _report(metrics={"terminal_error_m": 0.01})
    assert translate_report(report) == set()


def test_multiple_events_from_one_report():
    report = _report(
        skill="hold",
        outcome=SkillOutcome.SLIP,
        metrics={"stalled": True},
    )
    assert translate_report(report) == {
        SafetyEvent.WATCHDOG_STALL,
        SafetyEvent.OBJECT_DROPPED,
    }


# --------------------------------------------------------------------------- #
# Integration: the translation table actually reaches _safety_flags.
# --------------------------------------------------------------------------- #


def test_safety_flags_wires_stalled_into_watchdog_trip():
    stage_result = StageResult(
        stage=1,
        score=1,
        max_score=4,
        reports=[_report(metrics={"stalled": True})],
    )
    flags = _safety_flags(stage_result)
    assert flags == {
        "watchdog_trip": True,
        "collision": False,
        "dropped_object": False,
    }


def test_safety_flags_wires_slip_into_dropped_object():
    stage_result = StageResult(
        stage=4,
        score=1,
        max_score=4,
        reports=[_report(skill="hold", outcome=SkillOutcome.SLIP)],
    )
    flags = _safety_flags(stage_result)
    assert flags["dropped_object"] is True
    assert flags["watchdog_trip"] is False
    assert flags["collision"] is False


def test_safety_flags_all_clean_by_default():
    stage_result = StageResult(
        stage=1,
        score=4,
        max_score=4,
        reports=[_report(), _report(skill="grasp")],
    )
    assert _safety_flags(stage_result) == {
        "watchdog_trip": False,
        "collision": False,
        "dropped_object": False,
    }


# --------------------------------------------------------------------------- #
# fully_completed
# --------------------------------------------------------------------------- #


def test_fully_completed_true_when_all_objects_scored():
    r = StageResult(stage=1, score=4, max_score=4)
    assert r.completed is True
    assert r.fully_completed is True


def test_fully_completed_false_on_partial_credit():
    r = StageResult(stage=1, score=1, max_score=4)
    assert r.completed is True  # reached the stage
    assert r.fully_completed is False  # but not every object


def test_neither_completed_nor_fully_completed_off_exception_path():
    """A caught exception/timeout must never report completion of any
    kind -- mirrors how orchestrator._run_stage_isolated actually
    constructs a StageResult on that path (score=0, failure_reason set)."""
    r = StageResult(
        stage=2,
        score=0,
        max_score=config.STAGE_MAX_SCORE,
        reports=[],
        details={},
        failure_reason="TimeoutError: stage 2 exceeded its wall-clock budget",
    )
    assert r.completed is False
    assert r.fully_completed is False


def test_fully_completed_false_when_max_score_is_zero():
    r = StageResult(stage=3, score=0, max_score=0)
    assert r.fully_completed is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
