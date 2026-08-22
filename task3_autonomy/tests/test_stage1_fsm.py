# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from task3_autonomy.stage1_fsm import (
    Stage1Action,
    Stage1FSM,
    Stage1FsmConfig,
    Stage1Observation,
    Stage1State,
)


def _milestones() -> list[Stage1Observation]:
    return [
        Stage1Observation(at_pickup=True),
        Stage1Observation(tray_held=True),
        Stage1Observation(at_dining=True),
        Stage1Observation(placed=True),
        Stage1Observation(released=True),
        Stage1Observation(retreated=True),
    ]


def test_stage1_advances_in_order_and_completes() -> None:
    fsm = Stage1FSM()
    actions = [fsm.step(observation, 0.1) for observation in _milestones()]

    assert actions == [
        Stage1Action.GRASP_TRAY,
        Stage1Action.TRANSPORT_TO_DINING,
        Stage1Action.PLACE_OBJECTS,
        Stage1Action.RELEASE_TRAY,
        Stage1Action.RETREAT,
        Stage1Action.HOLD,
    ]
    assert fsm.succeeded
    assert fsm.history == [
        Stage1State.NAVIGATE_PICKUP,
        Stage1State.GRASP_TRAY,
        Stage1State.TRANSPORT,
        Stage1State.PLACE,
        Stage1State.RELEASE,
        Stage1State.RETREAT,
        Stage1State.COMPLETE,
    ]


def test_timeout_retries_once_then_aborts() -> None:
    fsm = Stage1FSM(Stage1FsmConfig(state_timeout_s=1.0, max_retries=1))

    assert (
        fsm.step(Stage1Observation(), 1.1) is Stage1Action.NAVIGATE_TO_PICKUP
    )
    assert fsm.retries == 1
    assert fsm.step(Stage1Observation(), 1.1) is Stage1Action.ABORT
    assert fsm.state is Stage1State.FAILED


def test_collision_aborts_without_spending_retry() -> None:
    fsm = Stage1FSM(Stage1FsmConfig(max_retries=3))

    assert (
        fsm.step(Stage1Observation(collision=True), 0.0) is Stage1Action.ABORT
    )
    assert fsm.state is Stage1State.FAILED
    assert fsm.retries == 0


def test_invalid_config_and_dt_are_rejected() -> None:
    with pytest.raises(ValueError):
        Stage1FsmConfig(state_timeout_s=0.0)
    with pytest.raises(ValueError):
        Stage1FsmConfig(max_retries=-1)
    with pytest.raises(ValueError):
        Stage1FSM().step(Stage1Observation(), -0.1)
