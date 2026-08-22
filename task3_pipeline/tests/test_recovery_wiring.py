# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""`recovery.py` wired into the retry loop (REV20 P4.5).

`recovery.py` and `tracked_object.py` were both fully implemented, tested, and
recorded DONE while being imported by nothing outside their own test files --
the "wired but disconnected" defect class that also hid `min_creep_mps` and the
missing cuRobo install. These tests exist so that connection cannot silently
come undone again: each one fails if `SelfCorrectingSkill.run` stops consulting
`recovery.decide`.
"""

from __future__ import annotations

from task3_pipeline.memory import ParamMemory
from task3_pipeline.outcomes import SkillOutcome
from task3_pipeline.policy import RetryPolicy
from task3_pipeline.skills import SelfCorrectingSkill


class _World:
    head_placement = "-"


def _runner() -> SelfCorrectingSkill:
    mem = ParamMemory()
    return SelfCorrectingSkill(_World(), mem, RetryPolicy(mem))


def _timeout_metrics(_params: dict) -> dict:
    # classify_hold is the only classifier that produces TIMEOUT: held for
    # less than the required duration, with no slip.
    return {"z_drop_m": 0.0, "held_seconds": 0.4, "required_seconds": 2.0}


# ---- the perception gate runs BEFORE the motion ------------------------- #


def test_stale_pose_reperceives_instead_of_acting():
    """P4.5's actual requirement: do not commit motion to a bad belief."""
    runner = _runner()
    calls: list[str] = []
    gate_results = [SkillOutcome.PERCEPTION_STALE, None]

    def gate() -> SkillOutcome | None:
        return gate_results.pop(0) if gate_results else None

    def reperceive() -> bool:
        calls.append("reperceive")
        return True

    def invoke(_p: dict) -> dict:
        calls.append("invoke")
        return {"terminal_error_m": 0.0}

    report = runner.run(
        "navigate",
        invoke,
        perception_gate=gate,
        reperceive=reperceive,
    )

    # Re-perception happened first, and the skill only ran once the belief
    # was usable -- not "ran, failed, retried".
    assert calls == ["reperceive", "invoke"]
    assert report.outcome is SkillOutcome.SUCCESS


def test_reperception_does_not_consume_the_retry_budget():
    """A flickering detector must not exhaust the budget without moving."""
    runner = _runner()
    runner.policy.budget = 1  # one real attempt only
    n_gate = {"n": 0}

    def gate() -> SkillOutcome | None:
        n_gate["n"] += 1
        # Stale for the first three passes, then fine.
        return SkillOutcome.PERCEPTION_STALE if n_gate["n"] <= 3 else None

    invoked: list[dict] = []

    def invoke(p: dict) -> dict:
        invoked.append(p)
        return {"terminal_error_m": 0.0}

    report = runner.run(
        "navigate",
        invoke,
        perception_gate=gate,
        reperceive=lambda: True,
        max_reperceive=5,
    )

    # Three free re-perceptions, then the single budgeted attempt still ran.
    assert len(invoked) == 1
    assert report.outcome is SkillOutcome.SUCCESS


def test_reperception_is_bounded_and_abandons_the_object():
    """An absent object must fail, not loop -- a loop reads as a hang."""
    runner = _runner()

    def invoke(_p: dict) -> dict:  # pragma: no cover - must never run
        raise AssertionError("skill ran despite an unusable belief")

    report = runner.run(
        "navigate",
        invoke,
        perception_gate=lambda: SkillOutcome.PERCEPTION_LOW_CONFIDENCE,
        reperceive=lambda: True,
        max_reperceive=2,
    )

    assert report.outcome is SkillOutcome.PERCEPTION_LOW_CONFIDENCE


def test_gate_with_no_way_to_refresh_does_not_act_on_the_bad_belief():
    runner = _runner()

    def invoke(_p: dict) -> dict:  # pragma: no cover - must never run
        raise AssertionError("skill ran despite an unusable belief")

    report = runner.run(
        "navigate",
        invoke,
        perception_gate=lambda: SkillOutcome.PERCEPTION_STALE,
    )

    assert report.outcome is SkillOutcome.PERCEPTION_STALE


# ---- outcomes after the motion ------------------------------------------ #


def test_timeout_abandons_the_object_instead_of_burning_another_budget():
    """Repeated full-budget timeouts are how episodes hit the wall clock
    with no EPISODE_RESULT. One attempt, then move on."""
    runner = _runner()
    runner.policy.budget = 3
    invoked: list[dict] = []

    def invoke(p: dict) -> dict:
        invoked.append(p)
        return _timeout_metrics(p)

    report = runner.run("hold", invoke)

    assert len(invoked) == 1, "a timeout should not be retried"
    assert report.outcome is SkillOutcome.TIMEOUT


def test_nav_short_replans_rather_than_retrying_grasp_params():
    runner = _runner()
    runner.policy.budget = 2
    replans: list[str] = []
    seq = [
        {"terminal_error_m": 0.5},  # NAV_SHORT
        {"terminal_error_m": 0.0},  # then good
    ]

    def invoke(_p: dict) -> dict:
        return seq.pop(0)

    def replan_nav() -> bool:
        replans.append("replan")
        return True

    report = runner.run("navigate", invoke, replan_nav=replan_nav)

    assert replans == ["replan"]
    assert report.outcome is SkillOutcome.SUCCESS


# ---- the default path is untouched -------------------------------------- #


def test_no_callbacks_means_behaviour_is_unchanged():
    """Every existing call site passes none of the new arguments. This is the
    regression guard for the GPU track's in-flight runs."""
    runner = _runner()
    runner.policy.budget = 2
    seq = [
        {"position_error_m": 9.0},  # reach fails
        {"position_error_m": 0.0, "strict_reach": True},  # then succeeds
    ]

    report = runner.run("reach", lambda _p: seq.pop(0))

    assert report.outcome is SkillOutcome.SUCCESS
    assert not seq, "both attempts should have run"


# ---- the real hold gate is actually consulted ---------------------------- #


def _hold_evidence(**over) -> dict:
    """A hold that passes the old check (no drop, enough time) -- the exact
    shape that used to be scored SUCCESS without any grasp evidence."""
    base = {
        "z_drop_m": 0.0,
        "held_seconds": 3.0,
        "required_seconds": 2.0,
        "gripper_position_rad": 0.075,  # inside the cage band
        "ee_pos_start": (0.0, 0.0, 0.0),
        "ee_pos_end": (0.0, 0.0, 0.10),
        "object_pos_start": (0.0, 0.0, 0.0),
        "object_pos_end": (0.0, 0.0, 0.10),
        "object_rise_m": 0.10,
    }
    base.update(over)
    return base


def test_hold_with_full_evidence_still_succeeds():
    from task3_pipeline.outcomes import classify

    outcome, _ = classify("hold", _hold_evidence())
    assert outcome is SkillOutcome.SUCCESS


def test_hold_fails_when_the_object_never_followed_the_gripper():
    """The object sat still while the EE moved away: not a hold, however
    long it was 'held' for."""
    from task3_pipeline.outcomes import classify

    outcome, diag = classify(
        "hold",
        _hold_evidence(
            object_pos_end=(0.0, 0.0, 0.0), object_rise_m=0.0
        ),
    )
    assert outcome is SkillOutcome.WEAK_GRASP
    assert "follow" in diag or "rise" in diag


def test_hold_fails_when_the_gripper_is_wide_open():
    from task3_pipeline.outcomes import classify

    outcome, diag = classify("hold", _hold_evidence(gripper_position_rad=0.8))
    assert outcome is SkillOutcome.WEAK_GRASP
    assert "cage band" in diag


def test_hold_without_evidence_keeps_the_old_behaviour():
    """A world that does not emit the grasp evidence must not be failed for
    an instrumentation gap."""
    from task3_pipeline.outcomes import classify

    outcome, _ = classify(
        "hold",
        {"z_drop_m": 0.0, "held_seconds": 3.0, "required_seconds": 2.0},
    )
    assert outcome is SkillOutcome.SUCCESS
