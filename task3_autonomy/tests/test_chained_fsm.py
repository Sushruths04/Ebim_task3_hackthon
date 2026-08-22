# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from task3_autonomy.chained_fsm import (
    ChainAction,
    ChainObservation,
    Task3ChainFSM,
    Task3Stage,
)


def test_chain_requires_each_stage_in_order() -> None:
    fsm = Task3ChainFSM()
    assert fsm.step(ChainObservation(stage4_complete=True)) is (
        ChainAction.RUN_STAGE1
    )
    assert fsm.step(ChainObservation(stage1_complete=True)) is (
        ChainAction.RUN_STAGE2
    )
    assert fsm.step(ChainObservation(stage2_complete=True)) is (
        ChainAction.RUN_STAGE3
    )
    assert fsm.step(ChainObservation(stage3_complete=True)) is (
        ChainAction.RUN_STAGE4
    )
    assert fsm.step(ChainObservation(stage4_complete=True)) is ChainAction.HOLD
    assert fsm.succeeded
    assert fsm.history == [
        Task3Stage.STAGE1,
        Task3Stage.STAGE2,
        Task3Stage.STAGE3,
        Task3Stage.STAGE4,
        Task3Stage.COMPLETE,
    ]


def test_chain_fails_closed_on_safety_event() -> None:
    fsm = Task3ChainFSM()
    assert fsm.step(ChainObservation(collision=True)) is ChainAction.ABORT
    assert fsm.stage is Task3Stage.FAILED
    assert (
        fsm.step(ChainObservation(stage1_complete=True)) is ChainAction.ABORT
    )
