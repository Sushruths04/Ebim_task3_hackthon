# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""SelfCorrectingSkill: the fast loop around one primitive.

    execute chosen params  ->  verify outcome  ->  if failed, ask the policy
    for the next params (using memory + the diagnosis)  ->  retry  ->  record
    everything to memory.

This is the single mechanism that replaces "human watches GIF, edits a
constant, reruns" -- applied uniformly to every skill in every stage. No
training. Returns the first SUCCESS, or the best partial attempt if the retry
budget is exhausted (partial points beat a hang).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from task3_pipeline import config, recovery
from task3_pipeline.memory import ParamMemory
from task3_pipeline.outcomes import SkillOutcome, SkillReport, classify
from task3_pipeline.policy import RetryPolicy

# invoke(params) -> raw metrics dict from the world
Invoke = Callable[[dict], dict]
# optional partial-credit reward in [0,1] from metrics (for memory ranking)
RewardFn = Callable[[dict], float]
# Returns a perception outcome to act on, or None when the belief is usable.
# Built from a TrackedObject's own is_stale/is_confident gates via
# `recovery.perception_outcome`.
PerceptionGate = Callable[[], SkillOutcome | None]
# Refresh the belief / ask the navigator to replan. True == it worked.
Refresh = Callable[[], bool]


@dataclass
class SelfCorrectingSkill:
    world: object
    memory: ParamMemory
    policy: RetryPolicy

    def run(
        self,
        skill: str,
        invoke: Invoke,
        *,
        object_name: str = "-",
        reward_fn: RewardFn | None = None,
        on_attempt: Callable[[SkillReport], None] | None = None,
        perception_gate: PerceptionGate | None = None,
        reperceive: Refresh | None = None,
        replan_nav: Refresh | None = None,
        max_reperceive: int = 2,
    ) -> SkillReport:
        head = getattr(self.world, "head_placement", "-")
        tried: list[dict] = []
        last: SkillReport | None = None
        best: SkillReport | None = None
        best_reward = -1.0
        reperceive_attempts = 0

        # Counted explicitly rather than by `range(budget + 1)` because a
        # re-perception is not an attempt at the task and must not consume an
        # iteration: with a fixed-length loop, a detector that flickers for a
        # few cycles would silently eat the whole budget without the robot
        # ever moving. `max_reperceive` is what bounds that path instead.
        attempts = 0
        while attempts <= self.policy.budget:
            # 2026-08-14: the retry loop is the second place a stage can run
            # away (budget=4 attempts, each a full motion). Checked here as
            # well as in stages._run so an abort does not have to wait for
            # the whole skill to exhaust its budget. No-op without a
            # deadline -- see config.check_stage_deadline.
            config.check_stage_deadline(self.world, f"retry skill={skill}")
            plan = self.policy.plan(
                skill, head_placement=head, object_name=object_name, last=last
            )
            params = next((p for p in plan if p not in tried), None)
            if params is None:
                break

            # REV20 P4.5: check the belief BEFORE committing motion to it.
            # Acting on a stale or untrusted pose aims a well-tuned motion at
            # the wrong place, so this gate runs ahead of `invoke` -- the skill
            # must not run at all, rather than run and be retried.
            gate_outcome = perception_gate() if perception_gate else None
            if gate_outcome is not None:
                decision = recovery.decide(
                    gate_outcome,
                    reperceive_attempts=reperceive_attempts,
                    max_reperceive=max_reperceive,
                )
                report = SkillReport(
                    skill, gate_outcome, dict(params), {}, decision.reason
                )
                if on_attempt:
                    on_attempt(report)
                if decision.action is recovery.RecoveryAction.REPERCEIVE:
                    reperceive_attempts += 1
                    if reperceive is not None and reperceive():
                        # Re-perception is not an attempt at the task, so it
                        # neither consumes the retry budget nor burns a param
                        # candidate: `params` is deliberately not appended to
                        # `tried`. A flickering detector must not exhaust the
                        # budget without the robot ever moving.
                        continue
                    # Nothing can refresh the belief, so re-deciding would
                    # loop on the same input. Treat it as the bounded case.
                    decision = recovery.decide(
                        gate_outcome,
                        reperceive_attempts=max_reperceive,
                        max_reperceive=max_reperceive,
                    )
                if decision.action is recovery.RecoveryAction.ABORT_OBJECT:
                    self.memory.save()
                    return best if best is not None else report
                # Anything else falls through to a normal attempt.

            tried.append(params)
            attempts += 1

            metrics = invoke(params)
            outcome, diag = classify(skill, metrics)
            report = SkillReport(
                skill, outcome, dict(params), dict(metrics), diag
            )
            reward = (
                1.0
                if report.ok
                else (reward_fn(metrics) if reward_fn else 0.0)
            )
            self.memory.record(
                report,
                reward=reward,
                head_placement=head,
                object_name=object_name,
            )
            if on_attempt:
                on_attempt(report)

            if reward > best_reward:
                best, best_reward = report, reward
            if report.ok:
                self.memory.save()
                return report

            decision = recovery.decide(
                report.outcome,
                reperceive_attempts=reperceive_attempts,
                max_reperceive=max_reperceive,
            )
            if decision.action is recovery.RecoveryAction.ABORT_OBJECT:
                # A skill that exhausted its step budget will usually exhaust
                # it again on the next parameter, which is how episodes reach
                # the wall clock with no EPISODE_RESULT. Give up on this
                # object so the episode can still score on the others.
                self.memory.save()
                return best if best is not None else report
            if decision.action is recovery.RecoveryAction.REPLAN_NAV:
                # A base that stopped short is a planning/control problem for
                # the navigator, not a grasp-parameter problem. Only skip the
                # parameter retry if something can actually replan.
                if replan_nav is not None and replan_nav():
                    continue
            last = report

        self.memory.save()
        return (
            best
            if best is not None
            else SkillReport(
                skill, SkillOutcome.TIMEOUT, {}, {}, "no attempts made"
            )
        )
