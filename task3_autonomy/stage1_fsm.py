# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded scripted FSM for Task 3 Stage 1 tray transport.

The state machine is simulator-independent. Isaac adapters report measured
milestones through :class:`Stage1Observation`; the controller only decides
which skill should run next and when a retry budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Stage1State(str, Enum):
    NAVIGATE_PICKUP = "navigate_pickup"
    GRASP_TRAY = "grasp_tray"
    TRANSPORT = "transport"
    PLACE = "place"
    RELEASE = "release"
    RETREAT = "retreat"
    COMPLETE = "complete"
    FAILED = "failed"


class Stage1Action(str, Enum):
    NAVIGATE_TO_PICKUP = "navigate_to_pickup"
    GRASP_TRAY = "grasp_tray"
    TRANSPORT_TO_DINING = "transport_to_dining"
    PLACE_OBJECTS = "place_objects"
    RELEASE_TRAY = "release_tray"
    RETREAT = "retreat"
    HOLD = "hold"
    ABORT = "abort"


@dataclass(frozen=True)
class Stage1Observation:
    """Measured milestone flags returned by a simulator adapter."""

    at_pickup: bool = False
    tray_held: bool = False
    at_dining: bool = False
    placed: bool = False
    released: bool = False
    retreated: bool = False
    collision: bool = False
    dropped: bool = False


@dataclass(frozen=True)
class Stage1FsmConfig:
    """Per-state budgets for one autonomous Stage 1 attempt."""

    state_timeout_s: float = 30.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        if self.state_timeout_s <= 0.0:
            raise ValueError("state_timeout_s must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass
class Stage1FSM:
    """Advance the Stage 1 skill sequence from measured observations."""

    cfg: Stage1FsmConfig = field(default_factory=Stage1FsmConfig)
    state: Stage1State = Stage1State.NAVIGATE_PICKUP
    elapsed_s: float = 0.0
    retries: int = 0
    history: list[Stage1State] = field(
        default_factory=lambda: [Stage1State.NAVIGATE_PICKUP]
    )

    _NEXT: dict[Stage1State, Stage1State] = field(
        default_factory=lambda: {
            Stage1State.NAVIGATE_PICKUP: Stage1State.GRASP_TRAY,
            Stage1State.GRASP_TRAY: Stage1State.TRANSPORT,
            Stage1State.TRANSPORT: Stage1State.PLACE,
            Stage1State.PLACE: Stage1State.RELEASE,
            Stage1State.RELEASE: Stage1State.RETREAT,
            Stage1State.RETREAT: Stage1State.COMPLETE,
        },
        init=False,
        repr=False,
    )

    _ACTIONS: dict[Stage1State, Stage1Action] = field(
        default_factory=lambda: {
            Stage1State.NAVIGATE_PICKUP: Stage1Action.NAVIGATE_TO_PICKUP,
            Stage1State.GRASP_TRAY: Stage1Action.GRASP_TRAY,
            Stage1State.TRANSPORT: Stage1Action.TRANSPORT_TO_DINING,
            Stage1State.PLACE: Stage1Action.PLACE_OBJECTS,
            Stage1State.RELEASE: Stage1Action.RELEASE_TRAY,
            Stage1State.RETREAT: Stage1Action.RETREAT,
        },
        init=False,
        repr=False,
    )

    def action(self) -> Stage1Action:
        """Return the skill command for the current state."""

        return self._ACTIONS.get(self.state, Stage1Action.HOLD)

    @property
    def done(self) -> bool:
        return self.state in (Stage1State.COMPLETE, Stage1State.FAILED)

    @property
    def succeeded(self) -> bool:
        return self.state is Stage1State.COMPLETE

    def step(
        self, observation: Stage1Observation, dt_s: float
    ) -> Stage1Action:
        """Consume one measured update and return the next skill command."""

        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        if self.done:
            return Stage1Action.HOLD if self.succeeded else Stage1Action.ABORT
        if observation.collision or observation.dropped:
            self.state = Stage1State.FAILED
            self.history.append(self.state)
            return Stage1Action.ABORT

        self.elapsed_s += dt_s
        if self._milestone_reached(observation):
            self._advance()
        elif self.elapsed_s > self.cfg.state_timeout_s:
            if self.retries < self.cfg.max_retries:
                self.retries += 1
                self.elapsed_s = 0.0
            else:
                self.state = Stage1State.FAILED
                self.history.append(self.state)
        return (
            self.action()
            if not self.done
            else (Stage1Action.HOLD if self.succeeded else Stage1Action.ABORT)
        )

    def _milestone_reached(self, observation: Stage1Observation) -> bool:
        return {
            Stage1State.NAVIGATE_PICKUP: observation.at_pickup,
            Stage1State.GRASP_TRAY: observation.tray_held,
            Stage1State.TRANSPORT: observation.at_dining,
            Stage1State.PLACE: observation.placed,
            Stage1State.RELEASE: observation.released,
            Stage1State.RETREAT: observation.retreated,
        }.get(self.state, False)

    def _advance(self) -> None:
        next_state = self._NEXT.get(self.state, Stage1State.COMPLETE)
        self.state = next_state
        self.elapsed_s = 0.0
        self.retries = 0
        self.history.append(next_state)
