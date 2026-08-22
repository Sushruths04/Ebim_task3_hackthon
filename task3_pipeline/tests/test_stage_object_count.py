# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REV12 T1: the scorer must count exactly the 4 real Stage-1/4 objects
(config.STAGE1_OBJECTS), not grading.py's own DEFAULT_STAGE1_OBJECTS (5
entries -- includes "simple_tray", a scene prop that is real+pushable but
does NOT count, per P6's GATE DEFINITIVE finding). Before this fix,
IsaacWorld.score_stage(1)/(4) called grading.score_stage1_table_setup /
score_stage4_cleanup WITHOUT object_names, so max_score silently came back
5 while only 4 objects were ever scored.

Run: python -m pytest task3_pipeline/tests/test_stage_object_count.py -q
  or: python -B task3_pipeline/tests/test_stage_object_count.py

Pure CPU -- no Isaac, no GPU. IsaacWorld is CPU-importable/constructible by
design (module docstring); this test never touches simulation_app.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # allow direct `python -B <file>` runs
    sys.path.insert(0, str(_REPO_ROOT))

from task3_pipeline import config  # noqa: E402
from task3_pipeline.world_isaac import IsaacWorld  # noqa: E402

# Fixed, arbitrary positions -- content doesn't matter, only that every
# STAGE1_OBJECTS name resolves to something so score_stage can run without
# an Isaac scene.
_FAKE_POSITIONS = {
    "plate2": (-3.0, 1.9, 0.7466),
    "cup": (-3.1, 1.95, 0.7466),
    "bowl2": (-3.2, 2.0, 0.7466),
    "spoon2": (-3.3, 1.85, 0.7466),
}


def _build_world():
    world = IsaacWorld.__new__(IsaacWorld)  # bypass __init__'s out_dir mkdir
    world.object_names = config.STAGE1_OBJECTS
    world.stage4_objects = None
    world.object_position = lambda name: _FAKE_POSITIONS[name]
    return world


def test_config_stage1_objects_is_exactly_four():
    assert len(config.STAGE1_OBJECTS) == 4, config.STAGE1_OBJECTS
    assert "simple_tray" not in config.STAGE1_OBJECTS


def test_grading_default_would_have_been_five():
    """Documents the defect this test guards against: grading.py's own
    default object list is NOT config.STAGE1_OBJECTS."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "evaluation" / "task3"))
    import grading  # noqa: E402

    assert len(grading.DEFAULT_STAGE1_OBJECTS) == 5
    assert "simple_tray" in grading.DEFAULT_STAGE1_OBJECTS


def test_score_stage1_max_score_is_four():
    world = _build_world()
    score, max_score, details = world.score_stage(1)
    assert max_score == 4, f"expected max_score 4, got {max_score}"
    assert details["object_names"] == list(config.STAGE1_OBJECTS)
    assert set(details["passed"]) | set(details["failed"]) == set(
        config.STAGE1_OBJECTS
    )


def test_score_stage4_max_score_is_four():
    world = _build_world()
    world.object_z = lambda name: _FAKE_POSITIONS[name][2]
    score, max_score, details = world.score_stage(4)
    assert max_score == 4, f"expected max_score 4, got {max_score}"
    assert details["object_names"] == list(config.STAGE1_OBJECTS)
    assert set(details["passed"]) | set(details["failed"]) == set(
        config.STAGE1_OBJECTS
    )


def test_score_stage_asserts_on_tampered_object_list(monkeypatch):
    world = _build_world()
    monkeypatch.setattr(config, "STAGE1_OBJECTS", ("cup", "bowl2"))
    try:
        raised = False
        try:
            world.score_stage(1)
        except AssertionError:
            raised = True
        assert raised, (
            "score_stage must assert when STAGE1_OBJECTS != 4 objects"
        )
    finally:
        pass


def test_total_across_four_stages_is_sixteen():
    assert config.STAGE_MAX_SCORE == 4
    assert config.STAGE_MAX_SCORE * 4 == 16


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
