# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REVIEW #9 (handoff sec 76) regression tests: the pipeline's own Stage-4
success predicate must agree with the OFFICIAL scorer, and Stage 4 must be
able to bound its work to a subset of objects without changing the score.

Run: python -m pytest task3_pipeline/tests/test_stage4_scoring.py -q
  or: python -B task3_pipeline/tests/test_stage4_scoring.py

Pure CPU -- no Isaac, no GPU.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # allow direct `python -B <file>` runs
    sys.path.insert(0, str(_REPO_ROOT))

from task3_pipeline import config  # noqa: E402
from task3_pipeline.memory import ParamMemory  # noqa: E402
from task3_pipeline.policy import RetryPolicy  # noqa: E402
from task3_pipeline.run_task3 import _parse_stage4_objects  # noqa: E402
from task3_pipeline.skills import SelfCorrectingSkill  # noqa: E402
from task3_pipeline.stages import plan_stage4  # noqa: E402
from task3_pipeline.world import MockWorld  # noqa: E402


def _load_grading():
    """Import the organizers' grading.py by path (it is not a package)."""
    path = _REPO_ROOT / "scripts" / "evaluation" / "task3" / "grading.py"
    spec = importlib.util.spec_from_file_location("_t3_grading", path)
    module = importlib.util.module_from_spec(spec)
    # Must be in sys.modules BEFORE exec: grading.py's dataclasses use
    # `from __future__ import annotations`, so their deferred annotations are
    # resolved by looking the module up by name.
    sys.modules["_t3_grading"] = module
    spec.loader.exec_module(module)
    return module


_CX, _CY = config.SINK_CENTER_XY
_Z = config.SINK_TABLETOP_Z

# (label, x, y, z, expected)
_CASES = [
    ("sink centre at exact tabletop z", _CX, _CY, _Z, True),
    # The two the OLD hand-picked tolerances wrongly accepted:
    ("5 cm below tabletop z", _CX, _CY, _Z - 0.05, False),
    ("0.4 m from sink centre", _CX + 0.4, _CY, _Z, False),
    # The observed sec-74 end state:
    ("on the floor", _CX, _CY, 0.0129, False),
    ("just inside x_max", config.SINK_BOUNDS["x_max"] - 1e-6, _CY, _Z, True),
    ("just outside x_max", config.SINK_BOUNDS["x_max"] + 1e-3, _CY, _Z, False),
    ("just outside y_min", _CX, config.SINK_BOUNDS["y_min"] - 1e-3, _Z, False),
]


def test_scores_in_sink_matches_expectations():
    for label, x, y, z, want in _CASES:
        got = config.scores_in_sink(x, y, z)
        assert got is want, f"{label}: got {got}, want {want}"


def test_scores_in_sink_agrees_with_official_scorer():
    """The whole point of REVIEW #9: never report a success the real scorer
    would reject. Cross-checked against grading.py itself, not a copy of its
    constants."""
    grading = _load_grading()
    for label, x, y, z, _want in _CASES:
        official = grading.score_stage4_cleanup(
            {"cup": grading.Bounds2D.from_point((x, y))},
            {"cup": z},
            ("cup",),
        )
        ours = config.scores_in_sink(x, y, z)
        assert (official.score == 1) is ours, (
            f"{label}: official={official.score} ours={ours}"
        )


def test_old_tolerances_were_genuinely_looser():
    """Guard the reason this fix exists: if someone reinstates the old
    hand-picked tolerances, this test says why that is wrong."""
    old_pass_but_really_fails = [
        (_CX, _CY, _Z - 0.05),  # old: z >= target_z - 0.05
        (_CX + 0.4, _CY, _Z),  # old: hypot(...) <= 0.5
    ]
    for x, y, z in old_pass_but_really_fails:
        old_gate = (
            (x - _CX) ** 2 + (y - _CY) ** 2
        ) ** 0.5 <= 0.5 and z >= _Z - 0.05
        assert old_gate, "case no longer exercises the old gate"
        assert not config.scores_in_sink(x, y, z)


def _build(objects=None):
    world = MockWorld(seed=42, head_placement="a")
    world.reset(seed=42, head_placement="a")
    if objects is not None:
        world.stage4_objects = objects
    memory = ParamMemory(os.path.join(tempfile.mkdtemp(), "m.json"))
    runner = SelfCorrectingSkill(
        world=world, memory=memory, policy=RetryPolicy(memory=memory)
    )
    return world, runner


def test_stage4_object_restriction_bounds_work_without_changing_score():
    """One object attempted => 1/4 of the skill calls, but max_score stays 4
    because score_stage(4) always scores the full set. This is what makes a
    single-object run legal AND cheap enough to finish inside the 3600 s
    ceiling (handoff sec 76). REVIEW #10 (handoff sec 86): reach() removed
    from the loop (navigate/grasp/cleanup remain), so it's 3 skill steps per
    object now, not 4."""
    world_all, runner_all = _build()
    res_all = plan_stage4(runner_all, world_all)

    world_one, runner_one = _build(("cup",))
    res_one = plan_stage4(runner_one, world_one)

    assert len(res_all.reports) == 12, "4 objects x 3 skill steps"
    assert len(res_one.reports) == 3, "1 object x 3 skill steps"
    assert res_all.max_score == res_one.max_score == 4


def test_stage4_objects_default_is_unchanged():
    world, runner = _build()
    assert getattr(world, "stage4_objects", None) is None
    assert len(plan_stage4(runner, world).reports) == 12


def test_parse_stage4_objects():
    assert _parse_stage4_objects(None) is None
    assert _parse_stage4_objects("") is None
    assert _parse_stage4_objects("cup") == ("cup",)
    assert _parse_stage4_objects(" cup , bowl2 ") == ("cup", "bowl2")


def test_parse_stage4_objects_rejects_unknown_names():
    """A typo must fail loudly at parse time, not silently run zero objects
    and report a legitimate-looking score 0."""
    try:
        _parse_stage4_objects("nonsense")
    except SystemExit:
        return
    raise AssertionError("unknown object name was not rejected")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    summary = (
        "all stage4 scoring tests passed."
        if not failures
        else f"{failures} FAILED"
    )
    print(f"\n{summary}")
    sys.exit(1 if failures else 0)
