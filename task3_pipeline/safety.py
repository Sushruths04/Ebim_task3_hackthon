# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Safety-vocabulary translation: real skill metrics -> the chain FSM's
typed safety events.

REV12 T3 fix: ``orchestrator._safety_flags`` used to read
``metrics.get("collision")`` / ``metrics.get("watchdog")`` /
``metrics.get("dropped")`` directly off each skill's raw metrics dict.
Nothing in this codebase ever sets those exact keys -- the real producers
are ``world_isaac.py``'s ``stalled`` flag (``ProgressWatchdog``, set on
``navigate``/``turn`` results), the ``z_drop_m`` measurement that
``outcomes.classify_hold`` already turns into ``SkillOutcome.SLIP``, and
``peak_head_force_n`` (Stage 2's ``feed_hold``, checked against
``config.HEAD_MAX_FORCE_N``). So the fail-closed chain's safety gate was
silently dead on every real run: no skill's metrics dict has ever
contained a "collision"/"watchdog"/"dropped" key, so
``ChainObservation``'s ``watchdog_trip``/``collision``/``dropped_object``
were always False regardless of what actually happened in the episode.

This module is the ONE place that bridges producer vocabulary to
consumer vocabulary, as an explicit, individually testable table --
see ``task3_pipeline/tests/test_safety_translation.py``, one test per
entry.
"""

from __future__ import annotations

from enum import Enum

from task3_pipeline import config
from task3_pipeline.outcomes import SkillOutcome, SkillReport


class SafetyEvent(str, Enum):
    """Typed safety events a single skill's measurements can raise."""

    WATCHDOG_STALL = "watchdog_stall"
    OBJECT_DROPPED = "object_dropped"
    HEAD_CONTACT_FORCE_EXCEEDED = "head_contact_force_exceeded"


def translate_report(report: SkillReport) -> set[SafetyEvent]:
    """The explicit translation table. Each branch documents the exact
    real producer signal it reads and why."""
    events: set[SafetyEvent] = set()
    metrics = report.metrics

    # Producer: world_isaac.py's ProgressWatchdog sets metrics["stalled"]
    # on navigate()/turn() when sampled displacement stops progressing
    # (see world_isaac.py:695-696, 762-770). No SkillOutcome value reads
    # this today (classify_navigate only looks at terminal_error_m /
    # object_dist_m), so it must be read from raw metrics, not `.outcome`.
    if bool(metrics.get("stalled")):
        events.add(SafetyEvent.WATCHDOG_STALL)

    # Producer: outcomes.classify_hold already turns a z_drop_m measurement
    # past config.THRESHOLDS.slip_drop_m into SkillOutcome.SLIP -- reuse
    # that classification rather than re-deriving the same threshold here.
    if report.outcome is SkillOutcome.SLIP:
        events.add(SafetyEvent.OBJECT_DROPPED)

    # Producer: world_isaac.py's _head_contact_force_n() / feed_hold()
    # (Stage 2 only) sets metrics["peak_head_force_n"] -- the one real
    # ISO/TS 15066 force measurement in this codebase (config.py sec 1).
    # Skills that never measure head-contact force leave this key unset;
    # that correctly produces no event rather than a false "collision-free"
    # claim -- it means "not measured here", not "measured and clean".
    peak_force = metrics.get("peak_head_force_n")
    if peak_force is not None and float(peak_force) > config.HEAD_MAX_FORCE_N:
        events.add(SafetyEvent.HEAD_CONTACT_FORCE_EXCEEDED)

    return events


# Which ChainObservation boolean each SafetyEvent sets. Kept as an explicit
# table (not inlined into the caller) so it's independently readable/testable.
CHAIN_OBSERVATION_FIELD = {
    SafetyEvent.WATCHDOG_STALL: "watchdog_trip",
    SafetyEvent.OBJECT_DROPPED: "dropped_object",
    SafetyEvent.HEAD_CONTACT_FORCE_EXCEEDED: "collision",
}
