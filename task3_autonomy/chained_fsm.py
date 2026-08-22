# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed stage sequencing for one autonomous Task 3 episode.

This module deliberately contains no simulator, LLM, or transport imports.
The Isaac adapter supplies measured completion predicates and the controller
only decides which stage may run next.  That keeps chained execution
deterministic and makes it impossible for a later stage to start after a
watchdog or safety failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Task3Stage(str, Enum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    STAGE3 = "stage3"
    STAGE4 = "stage4"
    COMPLETE = "complete"
    FAILED = "failed"


class ChainAction(str, Enum):
    RUN_STAGE1 = "run_stage1"
    RUN_STAGE2 = "run_stage2"
    RUN_STAGE3 = "run_stage3"
    RUN_STAGE4 = "run_stage4"
    HOLD = "hold"
    ABORT = "abort"


@dataclass(frozen=True)
class ChainObservation:
    """One measured episode update from the simulator adapter."""

    stage1_complete: bool = False
    stage2_complete: bool = False
    stage3_complete: bool = False
    stage4_complete: bool = False
    watchdog_trip: bool = False
    collision: bool = False
    dropped_object: bool = False


@dataclass
class Task3ChainFSM:
    """Advance stages 1→2→3→4 only on explicit measured completion."""

    stage: Task3Stage = Task3Stage.STAGE1
    history: list[Task3Stage] = field(
        default_factory=lambda: [Task3Stage.STAGE1]
    )

    _NEXT: dict[Task3Stage, Task3Stage] = field(
        default_factory=lambda: {
            Task3Stage.STAGE1: Task3Stage.STAGE2,
            Task3Stage.STAGE2: Task3Stage.STAGE3,
            Task3Stage.STAGE3: Task3Stage.STAGE4,
            Task3Stage.STAGE4: Task3Stage.COMPLETE,
        },
        init=False,
        repr=False,
    )

    _ACTIONS: dict[Task3Stage, ChainAction] = field(
        default_factory=lambda: {
            Task3Stage.STAGE1: ChainAction.RUN_STAGE1,
            Task3Stage.STAGE2: ChainAction.RUN_STAGE2,
            Task3Stage.STAGE3: ChainAction.RUN_STAGE3,
            Task3Stage.STAGE4: ChainAction.RUN_STAGE4,
        },
        init=False,
        repr=False,
    )

    @property
    def done(self) -> bool:
        return self.stage in (Task3Stage.COMPLETE, Task3Stage.FAILED)

    @property
    def succeeded(self) -> bool:
        return self.stage is Task3Stage.COMPLETE

    def action(self) -> ChainAction:
        if self.stage is Task3Stage.FAILED:
            return ChainAction.ABORT
        return self._ACTIONS.get(self.stage, ChainAction.HOLD)

    def step(self, observation: ChainObservation) -> ChainAction:
        """Consume a state update and return the stage-level action.

        Safety failures are terminal. Completion flags for future stages are
        ignored until their own stage is active; this prevents stale flags
        from accidentally skipping stages after a reset or carry-over event.
        """

        if self.done:
            return ChainAction.HOLD if self.succeeded else ChainAction.ABORT
        if (
            observation.watchdog_trip
            or observation.collision
            or observation.dropped_object
        ):
            self.stage = Task3Stage.FAILED
            self.history.append(self.stage)
            return ChainAction.ABORT

        completed = {
            Task3Stage.STAGE1: observation.stage1_complete,
            Task3Stage.STAGE2: observation.stage2_complete,
            Task3Stage.STAGE3: observation.stage3_complete,
            Task3Stage.STAGE4: observation.stage4_complete,
        }.get(self.stage, False)
        if completed:
            self.stage = self._NEXT[self.stage]
            self.history.append(self.stage)
        return self.action()
