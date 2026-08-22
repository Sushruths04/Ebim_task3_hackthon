# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Task 3 end-to-end autonomous pipeline with an integrated self-correction loop.

This package is a *drop-in orchestration layer* on top of the proven skill
primitives in ``task3_autonomy`` (navigation, dual-arm control, chained FSM).
It does not replace them. What it adds is the piece the old workflow was
missing: an automated verify -> diagnose -> adjust -> retry -> remember loop
that runs without a human watching a GIF.

Public entry point: ``task3_pipeline.orchestrator.Task3Pipeline``.
See ``README.md`` for architecture and integration notes.
"""

from task3_pipeline.outcomes import SkillOutcome, SkillReport
from task3_pipeline.memory import ParamMemory
from task3_pipeline.policy import RetryPolicy

__all__ = ["SkillOutcome", "SkillReport", "ParamMemory", "RetryPolicy"]
