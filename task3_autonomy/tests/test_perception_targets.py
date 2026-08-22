# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for perception_targets.py's pure logic: ranking/fallback
selection and the ER response parser. The GPU-only pieces (bbox_grasp_target,
screen_candidate_ik) touch USD/Isaac and are exercised on GPU by
scripts/task3/probe_perception_targets.py, never here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task3_autonomy.perception_targets import (  # noqa: E402
    IkScreenResult,
    ScreenedCandidate,
    best_feasible,
    er_grasp_targets_batched,
)


def _candidate(
    source: str, rank: int, feasible_sides: list[str]
) -> ScreenedCandidate:
    cand = ScreenedCandidate(
        object="cup",
        source=source,
        rank=rank,
        xyz=(0.0, 0.0, 0.0),
        yaw_rad=0.0,
    )
    cand.sides = [
        IkScreenResult(
            side=side,
            ik_feasible=side in feasible_sides,
            target_norm_from_base_m=0.5,
        )
        for side in ("left", "right")
    ]
    return cand


def test_best_feasible_falls_down_the_ranked_list_on_rejection():
    er_infeasible = _candidate("er", 0, feasible_sides=[])
    bbox_feasible = _candidate("bbox", 1, feasible_sides=["right"])
    result = best_feasible([er_infeasible, bbox_feasible])
    assert result is bbox_feasible
    assert result.source == "bbox"


def test_best_feasible_prefers_er_when_both_feasible():
    er_feasible = _candidate("er", 0, feasible_sides=["left"])
    bbox_feasible = _candidate("bbox", 1, feasible_sides=["left", "right"])
    result = best_feasible([er_feasible, bbox_feasible])
    assert result is er_feasible


def test_best_feasible_returns_none_when_nothing_is_ik_feasible():
    er_infeasible = _candidate("er", 0, feasible_sides=[])
    bbox_infeasible = _candidate("bbox", 1, feasible_sides=[])
    assert best_feasible([er_infeasible, bbox_infeasible]) is None


def test_screened_candidate_best_feasible_norm_m_ignores_infeasible_sides():
    cand = ScreenedCandidate(
        object="cup", source="er", rank=0, xyz=(0.0, 0.0, 0.0), yaw_rad=0.0
    )
    cand.sides = [
        IkScreenResult(
            side="left", ik_feasible=False, target_norm_from_base_m=0.2
        ),
        IkScreenResult(
            side="right", ik_feasible=True, target_norm_from_base_m=0.6
        ),
    ]
    assert cand.any_side_feasible is True
    assert cand.best_feasible_norm_m == 0.6


def test_er_grasp_targets_batched_parses_and_dedupes_by_object():
    def fake_call_er(frame_path, prompt, api_keys):
        assert "cup" in prompt and "bowl2" in prompt
        return (
            [
                {
                    "object": "cup",
                    "point": [500.0, 500.0],
                    "grasp_angle_deg": 30.0,
                },
                {
                    "object": "bowl2",
                    "point": [400.0, 600.0],
                    "grasp_angle_deg": 0.0,
                },
                # A duplicate second entry for cup must not overwrite the
                # first.
                {
                    "object": "cup",
                    "point": [1.0, 1.0],
                    "grasp_angle_deg": 999.0,
                },
            ],
            1.23,
            "abcdef",
        )

    by_name, latency, key_suffix = er_grasp_targets_batched(
        "fake_frame.png", ["cup", "bowl2"], ["fake-key"], fake_call_er
    )
    assert set(by_name) == {"cup", "bowl2"}
    assert by_name["cup"]["point"] == [500.0, 500.0]
    assert latency == 1.23
    assert key_suffix == "abcdef"


def test_er_grasp_targets_batched_ignores_objects_not_requested():
    def fake_call_er(frame_path, prompt, api_keys):
        return (
            [{"object": "not_a_stage4_object", "point": [1.0, 1.0]}],
            0.5,
            "xxxxxx",
        )

    by_name, _, _ = er_grasp_targets_batched(
        "fake_frame.png", ["cup"], ["fake-key"], fake_call_er
    )
    assert by_name == {}
