# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""R9 T1: the VM A / VM B grasp interface contract, tested cold -- exactly
what VM B has to build against without asking VM A anything."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task3_autonomy.grasp_contract import (  # noqa: E402
    CandidateFile,
    GraspCandidate,
    GraspContractError,
    GraspMemoryEntry,
    RankedFile,
    RankedGrasp,
    append_grasp_memory,
    candidates_path,
    load_candidates,
    load_ranked,
    ranked_path,
    read_grasp_memory,
    save_ranked,
)

_CANDIDATE_RAW = {
    "object": "bowl2",
    "object_pose": [-3.9, -0.8, 0.74],
    "generated_utc": "2026-08-05T00:00:00Z",
    "candidates": [
        {
            "id": 0,
            "position": [-3.9, -0.8, 0.82],
            "yaw_rad": 0.79,
            "tilt_rad": 0.0,
            "source": "er",
            "label": "rim",
            "confidence": 0.8,
        },
        {
            "id": 1,
            "position": [-3.85, -0.75, 0.82],
            "yaw_rad": 1.2,
            "tilt_rad": 0.1,
            "source": "er",
            "label": "rim-alt",
            "confidence": 0.6,
        },
    ],
}


def test_candidate_file_round_trips_through_json():
    parsed = CandidateFile.from_json(_CANDIDATE_RAW)
    assert parsed.object == "bowl2"
    assert len(parsed.candidates) == 2
    assert parsed.candidates[0].position == (-3.9, -0.8, 0.82)
    back = json.loads(json.dumps(parsed.to_json()))
    assert back == CandidateFile.from_json(back).to_json()


def test_candidate_file_rejects_missing_field():
    bad = {k: v for k, v in _CANDIDATE_RAW.items() if k != "generated_utc"}
    with pytest.raises(GraspContractError, match="generated_utc"):
        CandidateFile.from_json(bad)


def test_candidate_file_rejects_empty_candidates():
    bad = {**_CANDIDATE_RAW, "candidates": []}
    with pytest.raises(GraspContractError, match="non-empty"):
        CandidateFile.from_json(bad)


def test_candidate_file_rejects_duplicate_ids():
    dup = dict(_CANDIDATE_RAW["candidates"][0])
    bad = {**_CANDIDATE_RAW, "candidates": [dup, dup]}
    with pytest.raises(GraspContractError, match="duplicate"):
        CandidateFile.from_json(bad)


def test_candidate_position_must_be_xyz():
    bad_candidate = {**_CANDIDATE_RAW["candidates"][0], "position": [1.0, 2.0]}
    bad = {**_CANDIDATE_RAW, "candidates": [bad_candidate]}
    with pytest.raises(GraspContractError, match="x, y, z"):
        CandidateFile.from_json(bad)


def test_load_candidates_round_trips_via_disk(tmp_path):
    path = candidates_path("bowl2", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_CANDIDATE_RAW))
    loaded = load_candidates("bowl2", tmp_path)
    assert loaded.object == "bowl2"
    assert loaded.candidates[1].label == "rim-alt"


def test_load_candidates_rejects_filename_object_mismatch(tmp_path):
    path = candidates_path("plate2", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_CANDIDATE_RAW))  # object field says bowl2
    with pytest.raises(GraspContractError, match="!="):
        load_candidates("plate2", tmp_path)


_RANKED_RAW = {
    "object": "bowl2",
    "ranked": [
        {
            "candidate_id": 0,
            "side": "left",
            "stance_xy": [-3.77, -0.82],
            "stance_yaw": 3.142,
            "feasible": True,
            "ik_margin": 0.13,
            "rank": 0,
        },
        {
            "candidate_id": 0,
            "side": "right",
            "stance_xy": [-3.77, -0.82],
            "stance_yaw": 0.0,
            "feasible": False,
            "ik_margin": -1.0,
            "rank": 1,
        },
    ],
}


def test_ranked_file_round_trips_through_json():
    parsed = RankedFile.from_json(_RANKED_RAW)
    assert parsed.feasible_sorted() == (
        RankedGrasp.from_json(_RANKED_RAW["ranked"][0]),
    )


def test_ranked_file_rejects_bad_side():
    bad_entry = {**_RANKED_RAW["ranked"][0], "side": "up"}
    bad = {**_RANKED_RAW, "ranked": [bad_entry]}
    with pytest.raises(GraspContractError, match="side"):
        RankedFile.from_json(bad)


def test_ranked_file_rejects_unsorted_rank():
    bad = {
        "object": "bowl2",
        "ranked": [
            {**_RANKED_RAW["ranked"][0], "rank": 1},
            {**_RANKED_RAW["ranked"][1], "rank": 0},
        ],
    }
    with pytest.raises(GraspContractError, match="sorted by rank"):
        RankedFile.from_json(bad)


def test_save_and_load_ranked_round_trips_via_disk(tmp_path):
    ranked_file = RankedFile.from_json(_RANKED_RAW)
    written = save_ranked(ranked_file, tmp_path)
    assert written == ranked_path("bowl2", tmp_path)
    loaded = load_ranked("bowl2", tmp_path)
    assert loaded == ranked_file


def test_bowl2_case_the_gate_this_contract_must_support():
    """The exact split rev 9's GATE T2 requires the ranker to reproduce:
    same ER pose, feasible on LEFT, infeasible on RIGHT. Not a ranker test
    -- just confirms the schema can represent that split without loss."""
    parsed = RankedFile.from_json(_RANKED_RAW)
    left = next(r for r in parsed.ranked if r.side == "left")
    right = next(r for r in parsed.ranked if r.side == "right")
    assert left.feasible is True
    assert right.feasible is False


_MEMORY_ENTRY_RAW = {
    "object": "bowl2",
    "candidate_id": 0,
    "side": "left",
    "stance_xy": [-3.77, -0.82],
    "stance_yaw": 3.142,
    "predicted_feasible": True,
    "object_follows_ee": True,
    "position_error_m": 0.0551,
    "utc": "2026-08-05T00:10:00Z",
    "source": "ik_feasibility_sweep",
}


def test_grasp_memory_entry_round_trips_through_json():
    parsed = GraspMemoryEntry.from_json(_MEMORY_ENTRY_RAW)
    assert parsed.object_follows_ee is True
    assert parsed.to_json() == _MEMORY_ENTRY_RAW


def test_grasp_memory_entry_defaults_source_when_absent():
    raw = {k: v for k, v in _MEMORY_ENTRY_RAW.items() if k != "source"}
    parsed = GraspMemoryEntry.from_json(raw)
    assert parsed.source == ""


def test_append_grasp_memory_is_append_only(tmp_path):
    path = tmp_path / "grasp_memory.jsonl"
    entry_a = GraspMemoryEntry.from_json(_MEMORY_ENTRY_RAW)
    entry_b = GraspMemoryEntry.from_json(
        {**_MEMORY_ENTRY_RAW, "side": "right", "object_follows_ee": False}
    )
    append_grasp_memory(entry_a, path)
    append_grasp_memory(entry_b, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    entries = list(read_grasp_memory(path))
    assert entries == [entry_a, entry_b]


def test_read_grasp_memory_on_missing_file_yields_nothing(tmp_path):
    assert list(read_grasp_memory(tmp_path / "does_not_exist.jsonl")) == []


def test_read_grasp_memory_skips_blank_lines(tmp_path):
    path = tmp_path / "grasp_memory.jsonl"
    entry = GraspMemoryEntry.from_json(_MEMORY_ENTRY_RAW)
    path.write_text("\n" + json.dumps(entry.to_json()) + "\n\n")
    assert list(read_grasp_memory(path)) == [entry]


def test_candidate_from_json_rejects_non_dict():
    with pytest.raises(GraspContractError, match="JSON object"):
        GraspCandidate.from_json([1, 2, 3])
