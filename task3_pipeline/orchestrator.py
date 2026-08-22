# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Task3Pipeline: the end-to-end autonomous controller.

It sequences stages 1 -> 2 -> 3 -> 4 through the existing fail-closed
``Task3ChainFSM`` (safety events are terminal), runs each stage's plan through
the self-correcting skill loop, aggregates the score, and emits one
``EPISODE_RESULT`` JSON line (same convention as the repo's integration tests).

Nothing here imports Isaac. Swap ``MockWorld`` for ``IsaacWorld`` and the same
orchestrator runs the real robot.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from task3_autonomy.chained_fsm import (
    ChainObservation,
    Task3ChainFSM,
    Task3Stage,
)
from task3_pipeline import config
from task3_pipeline.config import DEFAULT_MEMORY_PATH, STAGE_MAX_SCORE
from task3_pipeline.memory import ParamMemory
from task3_pipeline.policy import RetryPolicy
from task3_pipeline.safety import CHAIN_OBSERVATION_FIELD, translate_report
from task3_pipeline.skills import SelfCorrectingSkill
from task3_pipeline.stages import STAGE_PLANS, StageResult, official_spec_ready


@dataclass
class EpisodeResult:
    seed: int
    head_placement: str
    stages: list[StageResult] = field(default_factory=list)
    aborted_at: int | None = None
    # handoff.md sec 18.1c: every run must print these so no future budget
    # is ever chosen again without citing a MEASURED s_per_tick. tick_count
    # is 0 for MockWorld (no simulation ticks exist there), which makes
    # s_per_tick None rather than a fabricated number.
    tick_count: int = 0
    wall_time_seconds: float = 0.0

    @property
    def s_per_tick(self) -> float | None:
        if self.tick_count <= 0:
            return None
        return round(self.wall_time_seconds / self.tick_count, 4)

    @property
    def total(self) -> int:
        return sum(s.score for s in self.stages)

    @property
    def max_total(self) -> int:
        return sum(s.max_score for s in self.stages) or 16

    @property
    def pct(self) -> float:
        return round(self.total / self.max_total, 3)

    @property
    def highest_stage(self) -> int:
        completed = [s.stage for s in self.stages if s.completed]
        return max(completed) if completed else 0

    def as_json(self) -> str:
        return json.dumps(
            {
                "EPISODE_RESULT": True,
                "seed": self.seed,
                "head_placement": self.head_placement,
                "total_score": self.total,
                "max_score": self.max_total,
                "pct": self.pct,
                "highest_stage_completed": self.highest_stage,
                "aborted_at_stage": self.aborted_at,
                "tick_count": self.tick_count,
                "wall_time_seconds": self.wall_time_seconds,
                "s_per_tick": self.s_per_tick,
                "per_stage": [
                    {
                        "stage": s.stage,
                        "score": s.score,
                        "max": s.max_score,
                        "details": s.details,
                        "failure_reason": s.failure_reason,
                        # REV12 T3: completed (score>0, "reached this
                        # stage") and fully_completed (score==max_score,
                        # "every object/attempt scored") are distinct
                        # claims -- report both, plus the raw per-object
                        # score breakdown (details["passed"]/["failed"]
                        # when the scorer provides it), never just
                        # highest_stage_completed alone.
                        "completed": s.completed,
                        "fully_completed": s.fully_completed,
                        # P0.9: score is always the local dev-scorer's
                        # number (development_score makes that explicit);
                        # official_spec_ready separately marks whether it
                        # is ALSO defensible under the real organizer rules
                        # (config.py: grading.py is a dev helper, not the
                        # official scorer).
                        "development_score": s.score,
                        "official_spec_ready": official_spec_ready(
                            s.stage, s.details
                        ),
                    }
                    for s in self.stages
                ],
            },
            sort_keys=True,
        )


class Task3Pipeline:
    def __init__(
        self,
        world,
        *,
        memory_path: str | None = DEFAULT_MEMORY_PATH,
        retry_budget: int | None = None,
        stage_wallclock_budget_s: float | None = None,
        measured_s_per_tick: float | None = None,
    ):
        self.world = world
        self.memory = (
            ParamMemory.load(memory_path) if memory_path else ParamMemory()
        )
        self.policy = RetryPolicy(self.memory)
        if retry_budget is not None:
            self.policy.budget = retry_budget
        self.runner = SelfCorrectingSkill(world, self.memory, self.policy)
        # None (the default) means "derive the ceiling from
        # config.STAGE_TICK_BUDGETS x measured_s_per_tick per stage" (sec
        # 18.1/18.6 W0.5) instead of one flat wall-clock number for every
        # stage. An explicit value here (run_task3.py --stage-timeout-s)
        # overrides that derivation uniformly -- kept for the single-stage
        # isolating experiments (e.g. sec 4.65's `--order 1
        # --stage-timeout-s 1200`) that want one known number, not a
        # per-stage guess.
        self.stage_wallclock_budget_s = stage_wallclock_budget_s
        self.measured_s_per_tick = measured_s_per_tick

    def run_episode(
        self,
        *,
        seed: int,
        head_placement: str,
        order: tuple[int, ...] = (1, 2, 3, 4),
    ) -> EpisodeResult:
        started_at = time.time()
        ticks_before = getattr(self.world, "_tick_count", 0)
        self.world.reset(seed=seed, head_placement=head_placement)
        chain = Task3ChainFSM()
        result = EpisodeResult(seed=seed, head_placement=head_placement)

        for stage in order:
            stage_result = self._run_stage_isolated(stage)
            result.stages.append(stage_result)

            # Feed the fail-closed chain: a safety event in any skill is
            # terminal; otherwise mark this stage attempted and advance
            # (this includes a caught exception/timeout -- fault isolation
            # means "banked 0 and moved on", not "safety-aborted the chain").
            safety = _safety_flags(stage_result)
            obs = ChainObservation(
                **{f"stage{stage}_complete": True}, **safety
            )
            chain.step(obs)
            if chain.stage is Task3Stage.FAILED:
                result.aborted_at = stage
                break

        self.memory.save()
        # MockWorld has no _tick_count (no simulation ticks exist); getattr
        # defaults to 0, which correctly makes EpisodeResult.s_per_tick None
        # instead of reporting a fabricated rate for a world with no ticks.
        result.tick_count = (
            getattr(self.world, "_tick_count", 0) - ticks_before
        )
        result.wall_time_seconds = round(time.time() - started_at, 3)
        return result

    def _run_stage_isolated(self, stage: int) -> StageResult:
        """Run one stage plan such that it can never take the episode down.

        An uncaught exception (a missing world method, a bad IK call) or a
        stage that runs long must not cost the remaining stages -- Stage 4
        is this project's only proven point and it must still be attempted
        even if Stage 2 raises. Runs the plan on a daemon thread so a
        genuine deadlock in the calling process doesn't block forever; the
        stage is *reported* as timed out (0/max, `failure_reason`) as soon
        as the budget elapses, so scoring/progress is never held up.

        CORRECTION (handoff.md sec 4.64): an earlier version of this method
        left the timed-out thread running and immediately started the next
        stage's thread against the same ``self.world`` -- neither Isaac Sim
        nor ``IsaacWorld`` are thread-safe, and a real GPU run showed this
        corrupts everything downstream (two threads' `apply_twist()` calls
        overwrite each other every tick; the base commanded in two
        directions at once never moves at all). **This method now always
        waits for the thread to actually finish before returning**, even
        after reporting the timeout -- every skill call in `stages.py` is
        internally tick-range-bounded (`for _ in range(ticks)`, never a
        literal `while True`), so this wait is bounded in practice by the
        slowest in-flight skill, not unbounded. The reported score/timing
        for THIS stage is unaffected; what changes is that the NEXT stage
        is now guaranteed to start only once this one has truly stopped
        touching ``self.world``.
        """
        plan = STAGE_PLANS[stage]
        outcome: queue.Queue = queue.Queue(maxsize=1)

        def _target() -> None:
            try:
                outcome.put(("ok", plan(self.runner, self.world)))
            except Exception as exc:  # noqa: BLE001 - isolate any stage
                # The isolation contract only ever surfaces str(exc) in the
                # final StageResult/EPISODE_RESULT -- no traceback, so a bug
                # inside a stage plan is otherwise undiagnosable from the
                # log alone. Print it here, still inside the except block
                # where the real traceback is available.
                print(
                    f"STAGE_EXCEPTION stage={stage} "
                    f"traceback:\n{traceback.format_exc()}",
                    flush=True,
                )
                outcome.put(("error", exc))
            except BaseException as exc:  # noqa: BLE001
                # `except Exception` does NOT cover what actually kills this
                # worker. Measured, allobj_4: the thread died at the head-
                # camera capture and the episode reported only
                # "WorkerDiedError: ... died before put(), e.g. via an
                # uncaught BaseException" -- the exact case the clause above
                # cannot see. The run cost four minutes of GPU and produced
                # no diagnosis at all, which is the failure this handler
                # exists to end.
                #
                # DIAGNOSE ONLY -- deliberately no `outcome.put()` here.
                # The WorkerDiedError path is a tested contract
                # (test_worker_dying_without_reporting_does_not_hang_the_
                # episode): the main thread must never block on this queue,
                # and the episode must go on to the next stage. Reporting an
                # outcome from here would quietly replace that guarantee.
                # The bug was never the outcome, only the silence.
                print(
                    f"STAGE_BASE_EXCEPTION stage={stage} "
                    f"type={type(exc).__name__} "
                    f"traceback:\n{traceback.format_exc()}",
                    flush=True,
                )
                raise

        budget_s = (
            self.stage_wallclock_budget_s
            if self.stage_wallclock_budget_s is not None
            else config.stage_wallclock_ceiling_s(
                stage, self.measured_s_per_tick
            )
        )
        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        # 2026-08-14: this used to be a bare `worker.join(budget_s)`, which
        # left the MAIN thread blocked for the entire stage. Rendering and
        # every Replicator annotator read may only happen on the main thread
        # (world_isaac sec 20b -- a worker-thread call froze, because Kit's
        # app-update pump is a coroutine only the main thread's event loop
        # can drain), so with the main thread parked in join() nothing could
        # capture a frame mid-stage at all. That is why every perception
        # grasp point had to be precomputed at reset(), before the robot had
        # navigated anywhere -- a frozen pose by construction.
        #
        # Now the main thread supervises in short slices and services the
        # worker's capture requests between them. This is NOT two threads
        # inside self.world at once -- the contract this docstring insists on
        # still holds -- because the requesting worker is parked on its own
        # reply Event and touches nothing while the main thread renders. They
        # are strictly interleaved.
        #
        # `service_main_thread_requests` never raises (see its docstring): a
        # capture failure is handed back to the worker, which falls back. An
        # exception escaping here would kill the supervision loop and take
        # the episode with it.
        # getattr, because most callers here pass a fake/lightweight world
        # (every orchestrator unit test does) that has no capture service and
        # never wants one. Absent the method this loop is a plain join with
        # extra steps, which is exactly the previous behaviour.
        service = getattr(self.world, "service_main_thread_requests", None)
        supervise_slice_s = 0.05
        deadline = time.monotonic() + budget_s
        while worker.is_alive() and time.monotonic() < deadline:
            if service is not None:
                service()
            worker.join(supervise_slice_s)

        if worker.is_alive():
            reason = (
                f"TimeoutError: stage {stage} exceeded "
                f"{budget_s:.0f}s wall-clock budget"
            )
            # REVIEW #3 R3.6: the score (0) is already determined at this
            # point; the join() below can still run for a while (bounded by
            # the slowest in-flight skill, sec 4.64) and used to do so
            # silently, which read from the log as an unexplained hang.
            # Say so immediately so it is distinguishable from a stuck run.
            print(
                f"STAGE_TIMEOUT {{'stage': {stage}, 'budget_s': "
                f"{budget_s:.0f}, 'reason': {reason!r}, 'note': "
                "'score is 0, waiting for worker thread to actually stop "
                "before starting the next stage'}}",
                flush=True,
            )
            # sec 4.64: block until it truly stops before letting the next
            # stage touch self.world -- see docstring. Keep servicing capture
            # requests while doing so: a bare join() here would stop
            # answering them, and a worker parked on a reply Event would then
            # wait out its full timeout on every capture it attempts during
            # shutdown, turning a stage timeout into a much longer stall.
            while worker.is_alive():
                if service is not None:
                    service()
                worker.join(supervise_slice_s)
            return StageResult(stage, 0, STAGE_MAX_SCORE, [], {}, reason)

        # sec 19b W1.4: the worker already joined (it is not alive), so it
        # must have put exactly one item before returning -- UNLESS it died
        # between the try/except in _target() and the put() call (e.g. a
        # BaseException such as SystemExit, which threading.excepthook
        # silently swallows with no traceback). A bare outcome.get() would
        # then block the main thread forever with no diagnostic at all.
        # queue.Empty on a thread that has already joined can only mean
        # that.
        try:
            kind, payload = outcome.get(timeout=5.0)
        except queue.Empty:
            reason = (
                f"WorkerDiedError: stage {stage}'s worker thread finished "
                "without reporting a result (died before put(), e.g. via "
                "an uncaught BaseException)"
            )
            return StageResult(stage, 0, STAGE_MAX_SCORE, [], {}, reason)
        if kind == "ok":
            return payload
        reason = f"{type(payload).__name__}: {payload}"
        return StageResult(stage, 0, STAGE_MAX_SCORE, [], {}, reason)


def _safety_flags(stage_result: StageResult) -> dict:
    """Map any skill-level safety metric into the chain's terminal
    predicates, via ``task3_pipeline.safety``'s explicit translation table
    (REV12 T3 -- this used to read metrics keys ("collision"/"watchdog"/
    "dropped") that no producer in this codebase ever sets, so the
    fail-closed gate was silently dead on every real run). (In the mock
    these are always clean; on Isaac the verifier fills them.)"""
    flags = {
        "watchdog_trip": False,
        "collision": False,
        "dropped_object": False,
    }
    for report in stage_result.reports:
        for event in translate_report(report):
            flags[CHAIN_OBSERVATION_FIELD[event]] = True
    return flags
