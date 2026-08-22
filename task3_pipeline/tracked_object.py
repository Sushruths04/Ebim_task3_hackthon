# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""World-state record for one perceived object (REV20 P4.3).

This is the unit `/world_state` publishes and the supervisor reasons over. It
exists so that "where is the cup" has exactly one answer with an explicit
freshness and an explicit confidence, instead of a bare xyz tuple whose age
nobody tracks.

WHY FRESHNESS IS A FIRST-CLASS FIELD. This project's recurring failure mode is
acting on a belief that was true a while ago: the documented "fling" hazard
launches objects across the room, and a stale pose then aims the next grasp at
empty floor. REV20 P4.5 is explicit that the planner must trigger
**re-perception instead of acting** when perception is uncertain, so staleness
and confidence have to be queryable properties, not comments.

All logic here is pure and CPU-testable: no Isaac, no torch, no ROS.

TIME BASE -- READ THIS BEFORE USING `is_stale`. `stamp_s` is **sim time**, the
same clock Isaac publishes on `/clock` (~19.7 Hz, GPU-verified in P0.4), NOT
wall time. `time.time()` must never be passed in. The two differ by roughly the
Unix epoch, so mixing them does not produce a slightly-wrong answer -- it makes
every object look stale by ~1.7 billion seconds, or never stale at all,
depending on which way round the mistake goes. P1 established the tell: a real
sim-time stamp is a small integer (`header.stamp.sec: 8`), not a ~1.7e9 epoch.
`is_stale` asserts on this rather than trusting the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

# A sim clock starts near zero and a Task 3 episode is ~1800 s. Any stamp past
# this is overwhelmingly likely to be a wall-clock epoch that leaked in.
_MAX_PLAUSIBLE_SIM_TIME_S = 1_000_000.0


class TrackStatus(str, Enum):
    """Lifecycle of a track, so the planner can tell "gone" from "unseen"."""

    TENTATIVE = "tentative"  # seen too few times to act on
    CONFIRMED = "confirmed"  # actively observed, safe to act on
    COASTING = "coasting"  # not currently visible, pose is dead reckoned
    LOST = "lost"  # missing long enough that the pose is meaningless


@dataclass
class TrackedObject:
    """One object the perception stack believes exists.

    Fields follow REV20 P4.3. Only `object_id`, `label`, `position` and
    `stamp_s` are required; everything else has a defined, honest default so a
    detector that cannot estimate orientation or extent does not have to invent
    one.
    """

    object_id: str
    label: str
    # Metres, in `frame_id`. Defaults to `map` because the supervisor plans in
    # it; a detector publishing camera-frame poses must say so explicitly.
    position: tuple[float, float, float]
    stamp_s: float
    frame_id: str = "map"

    # Quaternion (x, y, z, w). Identity means "unknown/unestimated", which is
    # honest for a rotationally symmetric object like a cup.
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    # Diagonal position covariance (m^2). None means the detector reported no
    # uncertainty -- deliberately distinct from "zero uncertainty", which would
    # be a false claim of perfect knowledge.
    position_covariance: tuple[float, float, float] | None = None

    confidence: float = 0.0
    status: TrackStatus = TrackStatus.TENTATIVE

    # Oriented bounding-box full extents (m), from the P4.2 geometry pipeline
    # (depth crop -> plane removal -> clustering -> PCA -> oriented bbox).
    extent: tuple[float, float, float] | None = None

    # Whether the gripper can actually take this. Distinct from confidence: a
    # perfectly-perceived object can still be ungraspable. A Robotiq 2F-85
    # opens ~0.085 m, so extent alone does not settle it -- pose matters too.
    graspable: bool | None = None

    # Where this object is supposed to end up (e.g. the sink bounds for
    # Stage 4). None means "no goal assigned", not "already placed".
    target_location: tuple[float, float, float] | None = None

    # Whether the current stage cares about this object at all. Keeps
    # collateral objects (the ones the fling hazard scatters) in the world
    # model without letting them attract planning effort.
    task_relevant: bool = False

    observation_count: int = 1
    metadata: dict = field(default_factory=dict)

    # -- queries ---------------------------------------------------------- #

    def age_s(self, now_s: float) -> float:
        """Seconds since this track was last updated, in sim time."""
        _assert_sim_time(now_s, "now_s")
        _assert_sim_time(self.stamp_s, "stamp_s")
        return now_s - self.stamp_s

    def is_stale(self, now_s: float, max_age_s: float) -> bool:
        """True when the pose is too old to act on.

        A negative age (a stamp from the future) counts as stale rather than
        fresh: it means the clocks disagree, and acting on a belief whose
        provenance is incoherent is exactly what this class exists to prevent.
        """
        if max_age_s < 0.0:
            raise ValueError(f"max_age_s must be >= 0, got {max_age_s}")
        age = self.age_s(now_s)
        return age < 0.0 or age > max_age_s

    def is_confident(self, tau: float) -> bool:
        """True when confidence clears `tau` AND the track is actable.

        Confidence alone is not sufficient. A TENTATIVE track can carry a high
        per-detection score after a single frame, and a LOST track's last score
        says nothing about where the object is now. Both must be excluded, or
        `is_confident` silently means "was confident once".
        """
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        if self.status in (TrackStatus.TENTATIVE, TrackStatus.LOST):
            return False
        return self.confidence >= tau

    def is_actionable(self, now_s: float, max_age_s: float, tau: float) -> bool:
        """The single check a planner should call before acting on this pose."""
        return not self.is_stale(now_s, max_age_s) and self.is_confident(tau)

    def position_sigma_m(self) -> float | None:
        """Scalar 1-sigma position uncertainty, or None if unreported."""
        if self.position_covariance is None:
            return None
        return math.sqrt(sum(self.position_covariance))

    def distance_to(self, point: tuple[float, float, float]) -> float:
        return math.dist(self.position, point)


def _assert_sim_time(value: float, name: str) -> None:
    """Catch a wall-clock timestamp before it silently corrupts staleness.

    See the module docstring: mixing wall time and sim time here does not
    degrade gracefully, so this fails loudly instead.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0 (sim time), got {value!r}")
    if value > _MAX_PLAUSIBLE_SIM_TIME_S:
        raise ValueError(
            f"{name}={value!r} looks like a wall-clock epoch, not sim time. "
            "TrackedObject timestamps must come from Isaac's /clock. "
            "See the module docstring."
        )
