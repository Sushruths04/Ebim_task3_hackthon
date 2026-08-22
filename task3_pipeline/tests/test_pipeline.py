# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the Task 3 self-correcting pipeline.

Run: python -m pytest task3_pipeline/tests -q
  or: python -B task3_pipeline/tests/test_pipeline.py   (no pytest needed)

These prove the *logic* (verifier, memory, retry, orchestration) without Isaac.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import time
from pathlib import Path

from task3_autonomy.navigation import (
    STANCE_REACH_RADIUS_M,
    point_clears_island,
    route_avoiding_island,
)
from task3_pipeline import config
from task3_pipeline.memory import ParamMemory
from task3_pipeline.orchestrator import Task3Pipeline
from task3_pipeline.outcomes import SkillOutcome, SkillReport, classify
from task3_pipeline.policy import RetryPolicy
from task3_pipeline.seats import (
    TABLE_SEAT_POSITIONS,
    assigned_seats,
    object_to_seat,
)
from task3_pipeline.skills import SelfCorrectingSkill
from task3_pipeline.stages import (
    BIMANUAL_OBJECTS,
    StageResult,
    official_spec_ready,
    plan_stage1,
    plan_stage4,
)
from task3_pipeline.world import MockWorld


def _load_grading_module():
    """Import the organizers' pure-Python grading helpers by file path.

    ``scripts/evaluation/task3/grading.py`` is not part of an importable
    package (no ``__init__.py`` anywhere under ``scripts/``), so it is loaded
    directly like the organizers' own
    ``scripts/evaluation/task3/tests/test_grading.py`` does (via sys.path).
    It has zero Isaac imports (see its own module docstring), so this works
    on plain CPU Python.
    """
    repo_root = Path(__file__).resolve().parents[2]
    grading_path = (
        repo_root / "scripts" / "evaluation" / "task3" / "grading.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task3_grading_for_seat_tests", grading_path
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses' frozen(eq=True) processing looks the module up in
    # sys.modules by name, so it must be registered before exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_grading = _load_grading_module()


# ---- verifier ----------------------------------------------------------- #


def test_verifier_labels_weak_grasp():
    outcome, diag = classify("grasp", {"gripper_rad": 0.63, "contact": True})
    assert outcome is SkillOutcome.WEAK_GRASP
    assert "0.63" in diag


def test_verifier_labels_firm_grasp_success():
    # Honest SUCCESS requires proof of an actual hold, not just a closed cage.
    outcome, _ = classify(
        "grasp",
        {
            "gripper_rad": 0.076,
            "contact": True,
            "object_follows_ee": True,
            "object_ee_dist_m": 0.01,
        },
    )
    assert outcome is SkillOutcome.SUCCESS


def test_verifier_flags_ik_fail_on_bad_reach():
    outcome, _ = classify(
        "reach", {"position_error_m": 0.079, "strict_reach": False}
    )
    assert outcome is SkillOutcome.IK_FAIL


def test_verifier_navigate_short_of_target_but_object_in_reach_is_success():
    # ACTIVE_BRIEF.md sec 3.5/T3: a navigate that stops 0.83 m short of its
    # nominal target, but leaves the object only 0.74 m from the achieved
    # base pose (inside the reach envelope), is a real success -- raw
    # distance-to-target alone previously turned usable stops into
    # NAV_SHORT, feeding a retry storm even though pregrasp/descend went
    # on to succeed from that exact pose.
    outcome, diag = classify(
        "navigate", {"terminal_error_m": 0.83, "object_dist_m": 0.74}
    )
    assert outcome is SkillOutcome.SUCCESS
    assert "0.74" in diag


def test_verifier_navigate_short_and_object_out_of_reach_is_nav_short():
    outcome, _ = classify(
        "navigate", {"terminal_error_m": 0.83, "object_dist_m": 1.6}
    )
    assert outcome is SkillOutcome.NAV_SHORT


def test_verifier_navigate_success_without_object_dist_metric_unaffected():
    # Callers that never pass reach_check_object (older code paths) must
    # keep the original tight-tolerance behaviour.
    outcome, _ = classify("navigate", {"terminal_error_m": 0.02})
    assert outcome is SkillOutcome.SUCCESS
    outcome, _ = classify("navigate", {"terminal_error_m": 0.83})
    assert outcome is SkillOutcome.NAV_SHORT


def test_verifier_grasp_closed_on_empty_air_is_weak_not_success():
    # The recurring project bug: gripper cage angle looks tight (below the
    # cage threshold) and contact was reported, but the object is NOT
    # following the end-effector and is far away -- this must NOT be
    # classified SUCCESS.
    outcome, diag = classify(
        "grasp",
        {
            "gripper_rad": 0.076,
            "contact": True,
            "object_follows_ee": False,
            "object_ee_dist_m": 0.22,
        },
    )
    assert outcome is SkillOutcome.WEAK_GRASP
    assert "not held" in diag
    assert "0.076" in diag


def test_verifier_grasp_missing_hold_evidence_is_weak_not_success():
    # No object_follows_ee / object_ee_dist_m supplied at all -- a closed
    # cage alone is not proof of a hold, so this must not default to SUCCESS.
    outcome, _ = classify("grasp", {"gripper_rad": 0.076, "contact": True})
    assert outcome is SkillOutcome.WEAK_GRASP


# ---- memory ------------------------------------------------------------- #


def test_memory_roundtrip_and_best_params(tmp_path=None):
    path = (
        (tmp_path or tempfile.mkdtemp()).__str__() + "/mem.json"
        if tmp_path
        else tempfile.mktemp(suffix=".json")
    )
    mem = ParamMemory(path=path)
    mem.record(
        SkillReport(
            "grasp", SkillOutcome.WEAK_GRASP, {"approach_stance": "east"}
        ),
        reward=0.2,
        object_name="cup",
    )
    mem.record(
        SkillReport(
            "grasp", SkillOutcome.SUCCESS, {"approach_stance": "north"}
        ),
        reward=1.0,
        object_name="cup",
    )
    mem.save()
    reloaded = ParamMemory.load(path)
    assert reloaded.best_params("grasp", object_name="cup") == {
        "approach_stance": "north"
    }
    assert {"approach_stance": "east"} in reloaded.failed_params(
        "grasp", object_name="cup"
    )


# ---- policy ------------------------------------------------------------- #


def test_policy_flips_stance_after_ik_fail():
    mem = ParamMemory()
    pol = RetryPolicy(mem)
    last = SkillReport(
        "grasp", SkillOutcome.IK_FAIL, {"approach_stance": "east"}
    )
    plan = pol.plan("grasp", object_name="cup", last=last)
    # The first candidate should NOT keep the failing stance.
    assert plan[0].get("approach_stance") == "north"


# ---- self-correcting skill --------------------------------------------- #


def test_skill_recovers_from_ik_fail_via_retry():
    world = MockWorld(seed=1)
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))
    # grasp defaults to east (fails); loop must find north and succeed.
    report = runner.run(
        "grasp",
        lambda p: world.grasp("right", "cup", **p),
        object_name="cup",
        reward_fn=lambda m: max(0.0, 1 - m["gripper_rad"] / 0.8),
    )
    assert report.outcome is SkillOutcome.SUCCESS
    assert report.params.get("approach_stance") == "north"


# ---- seats --------------------------------------------------------------- #


def test_assigned_seats_returns_distinct_seats_inside_dining_area():
    dining_area = _grading.TASK3_DINING_AREA
    seats = assigned_seats(seed=None)
    assert len(seats) == 3
    assert len({s.seat_id for s in seats}) == 3  # distinct
    for seat in seats:
        assert seat.seat_id in TABLE_SEAT_POSITIONS
        assert dining_area.contains_xy((seat.x, seat.y))


def test_assigned_seats_seeded_is_deterministic_and_distinct():
    seats_a = assigned_seats(seed=42, count=4)
    seats_b = assigned_seats(seed=42, count=4)
    assert [s.seat_id for s in seats_a] == [s.seat_id for s in seats_b]
    assert len({s.seat_id for s in seats_a}) == 4


def test_object_to_seat_targets_classify_as_dining_per_shipped_scorer():
    # This proves that placing the 4 real Stage-1 objects at their assigned
    # seat targets passes the ONLY real scorer that ships anywhere
    # (grading.py's dining-rectangle classifier) -- the local Stage-1
    # validation gate for T1.
    seats = assigned_seats(seed=None)
    mapping = object_to_seat(list(config.STAGE1_OBJECTS), seats)
    assert set(mapping) == set(config.STAGE1_OBJECTS)
    for obj, seat in mapping.items():
        area = _grading.classify_table_area((seat.x, seat.y))
        assert area == "dining", (
            f"{obj} -> seat {seat.seat_id} classified as {area!r}"
        )


# ---- end-to-end orchestration ------------------------------------------ #


def test_full_episode_reaches_70pct():
    world = MockWorld(seed=7, head_placement="a")
    pipe = Task3Pipeline(world, memory_path=None)
    result = pipe.run_episode(seed=7, head_placement="a")
    assert result.max_total == 16
    assert result.highest_stage == 4  # every stage attempted
    assert result.pct >= 0.70, result.as_json()  # >= 11/16


def test_plan_stage1_targets_4_objects_no_tray():
    # Real Stage 1 (organizer prose rules): 4 objects (plate, cup, bowl+beans,
    # spoon), NO tray, carried individually to assigned seats.
    world = MockWorld(seed=3, head_placement="a")
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    # The SET and the count are what this test is about -- 4 real objects,
    # no tray, per the organizer prose rules. The order is deliberately not
    # asserted: it was changed 2026-08-14 to attempt the graspable objects
    # before `plate2`, which a parallel jaw cannot close on and which used
    # to consume most of the stage budget before anything else was tried
    # once. Pinning the sequence here would make that an unrelated test
    # failure every time the attempt order is tuned.
    assert set(config.STAGE1_OBJECTS) == {"plate2", "cup", "bowl2", "spoon2"}
    assert len(config.STAGE1_OBJECTS) == 4
    assert "simple_tray" not in config.STAGE1_OBJECTS

    result = plan_stage1(runner, world)

    # The per-object loop runs navigate/reach/grasp/cleanup for each of the 4
    # real objects (4 skills * 4 objects = 16 reports minimum; retries only
    # add more) and must never touch "simple_tray".
    assert len(result.reports) >= 16
    for obj in config.STAGE1_OBJECTS:
        assert obj != "simple_tray"
        recorded = [k for k in mem._store if k.endswith(f"|{obj}")]
        assert recorded, f"expected memory entries recorded for object {obj}"
    assert "simple_tray" not in "".join(mem._store.keys())
    assert result.score >= 0


class _NavTargetRecordingWorld(MockWorld):
    """Records every navigate_to() target (and the object it was aimed at,
    at call time) so plan_stage1/plan_stage4 can be checked against the
    island bbox and the reach budget without needing Isaac."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.nav_calls: list[
            tuple[str | None, tuple[float, float] | None, tuple[float, float]]
        ] = []

    def navigate_to(self, x, y, yaw=None, **p):
        obj = p.get("reach_check_object")
        obj_xy = self.object_xy(obj) if obj is not None else None
        self.nav_calls.append((obj, obj_xy, (x, y)))
        return super().navigate_to(x, y, yaw, **p)


def test_plan_stage1_navigate_target_never_inside_island_footprint():
    # ACTIVE_BRIEF.md T1: Stage 1's navigate step must target a reach-safe
    # stance, never the raw object xy -- every real Stage 1/4 stall traced
    # to a navigate target sitting inside the kitchen island.
    world = _NavTargetRecordingWorld(seed=3, head_placement="a")
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    plan_stage1(runner, world)

    assert world.nav_calls, "expected at least one navigate_to call"
    for _, _, xy in world.nav_calls:
        assert point_clears_island(xy), xy


# ---------------------------------------------------------------------- #
# REV12 T6: plan_stage1 uses _select_arm_side (T3) instead of a hardcoded
# "right" literal, and reach_and_grasp_ranked's "no ranked file" fallback
# gives plan_stage1 the identical reach()/grasp() call sequence as the
# pre-T6 hardcoded-"right" path.
# ---------------------------------------------------------------------- #


class _ArmSideRecordingWorld(MockWorld):
    """MockWorld (no reach_and_grasp_ranked) with a forced, non-"right"
    _select_arm_side -- proves plan_stage1 actually consults the method
    rather than hardcoding "right", using the plain reach()/grasp() path
    every MockWorld-based test already exercises."""

    def __init__(self, *a, forced_side="left", **kw):
        super().__init__(*a, **kw)
        self.forced_side = forced_side
        self.reach_sides: list[str] = []
        self.grasp_sides: list[str] = []

    def _select_arm_side(self, object_name):
        return self.forced_side

    def reach(self, side, object_name, **p):
        self.reach_sides.append(side)
        return super().reach(side, object_name, **p)

    def grasp(self, side, object_name, **p):
        self.grasp_sides.append(side)
        return super().grasp(side, object_name, **p)


def test_plan_stage1_uses_select_arm_side_not_literal_right():
    world = _ArmSideRecordingWorld(
        seed=3, head_placement="a", forced_side="left"
    )
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    plan_stage1(runner, world)

    assert world.reach_sides, "expected at least one reach() call"
    assert world.grasp_sides, "expected at least one grasp() call"
    assert set(world.reach_sides) == {"left"}
    assert set(world.grasp_sides) == {"left"}
    assert "right" not in world.reach_sides + world.grasp_sides


def test_plan_stage1_falls_back_to_right_without_select_arm_side():
    """MockWorld has no _select_arm_side -- must still get the historical
    "right" default, not crash or silently pick something else."""
    world = MockWorld(seed=3, head_placement="a")
    assert not hasattr(world, "_select_arm_side")
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    result = plan_stage1(runner, world)

    assert result.score >= 0


def test_plan_stage1_ranked_grasp_no_file_matches_old_call_sequence():
    """GATE (ii): an object with no ranked candidates file must give
    plan_stage1 the IDENTICAL reach()/grasp() call sequence the old
    hardcoded-"right" path produced -- via the real
    IsaacWorld.reach_and_grasp_ranked fallback, not a reimplementation.

    2026-08-16: cup is now routed through reach_bimanual()/grasp_bimanual()
    instead (stages.py's BIMANUAL_OBJECTS), a deliberate behavior change --
    GPU-verified n=3, real navigation, the first real end-to-end lift in
    this project's history (see world_isaac.py's reach_bimanual
    docstring). The other 3 STAGE1_OBJECTS are unaffected and still take
    the single-arm path this test was written to pin."""
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._select_arm_side = lambda object_name: "right"
    world._load_ranked_grasp_plan = lambda object_name, side: ([], None)
    world._stance_for = lambda object_xy, approach: ((-4.0, -1.0), 0.0)
    world.object_xy = lambda name: (-4.0, -1.0)
    world.navigate_to = lambda *a, **kw: {"terminal_error_m": 0.01}
    world.carry_object_to = lambda *a, **kw: {"scored": True}
    world.score_stage = lambda stage: (4, 4, {"passed": [], "failed": []})

    reach_calls: list[dict] = []
    grasp_calls: list[dict] = []
    reach_bimanual_calls: list[dict] = []
    grasp_bimanual_calls: list[dict] = []

    def _fake_reach(side, object_name, **p):
        reach_calls.append({"side": side, "object": object_name, **p})
        return {"position_error_m": 0.02}

    def _fake_grasp(side, object_name, **p):
        grasp_calls.append({"side": side, "object": object_name, **p})
        return {"object_follows_ee": True}

    def _fake_reach_bimanual(object_name, **p):
        reach_bimanual_calls.append({"object": object_name, **p})
        return {"ok": True}

    def _fake_grasp_bimanual(object_name, **p):
        grasp_bimanual_calls.append({"object": object_name, **p})
        return {"held": True, "scored": True}

    world.reach = _fake_reach
    world.grasp = _fake_grasp
    world.reach_bimanual = _fake_reach_bimanual
    world.grasp_bimanual = _fake_grasp_bimanual
    world._held = None  # neither fake grasp path sets it; no lift_bimanual call

    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))
    plan_stage1(runner, world)

    # 3 non-bimanual objects (spoon2, bowl2, plate2), 1 reach + 1 grasp
    # each (no-ranked-file fallback is a single attempt, same as the old
    # direct reach()+grasp() pair) -- no stance/grasp_xyz overrides,
    # byte-identical to calling reach()/grasp() directly (mirrors
    # test_reach_and_grasp_ranked_falls_back_to_hardcoded_when_no_ranked_plan).
    # 2026-08-21: the counts are DERIVED from BIMANUAL_OBJECTS rather than
    # hardcoded. This test pinned "3 single-arm + cup bimanual" from when
    # that set was {"cup"}; it has since grown to {"cup", "bowl2",
    # "plate2"} (a deliberate change -- see the set's own comment), and the
    # test went red for tracking a number instead of the invariant. The
    # invariant is the split itself: every object in the set takes the
    # bimanual pair, every object outside it takes the single-arm pair with
    # no stance/grasp overrides.
    single_arm = [o for o in config.STAGE1_OBJECTS if o not in BIMANUAL_OBJECTS]
    bimanual = [o for o in config.STAGE1_OBJECTS if o in BIMANUAL_OBJECTS]

    assert len(reach_calls) == len(single_arm)
    assert len(grasp_calls) == len(single_arm)
    assert [c["object"] for c in reach_calls] == single_arm
    for call in reach_calls:
        assert call["side"] == "right"
        assert "grasp_xyz_override" not in call
        assert "stance_xy_override" not in call
    for call in grasp_calls:
        assert call["side"] == "right"

    assert len(reach_bimanual_calls) == len(bimanual)
    assert len(grasp_bimanual_calls) == len(bimanual)
    assert [c["object"] for c in reach_bimanual_calls] == bimanual
    assert [c["object"] for c in grasp_bimanual_calls] == bimanual


def test_plan_stage4_grasp_uses_select_arm_side_not_literal_right():
    world = _ArmSideRecordingWorld(
        seed=3, head_placement="a", forced_side="left"
    )
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    plan_stage4(runner, world)

    assert world.grasp_sides, "expected at least one grasp() call"
    assert set(world.grasp_sides) == {"left"}
    # REVIEW #10: plan_stage4 deliberately has no reach() step -- confirm
    # this fix didn't reintroduce one.
    assert world.reach_sides == []


def test_plan_stage4_navigate_target_never_inside_island_footprint():
    world = _NavTargetRecordingWorld(seed=3, head_placement="a")
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    plan_stage4(runner, world)

    assert world.nav_calls, "expected at least one navigate_to call"
    for _, _, xy in world.nav_calls:
        assert point_clears_island(xy), xy


def test_plan_stage4_navigate_reach_budget_preserved():
    # T2: only the *placement* is corrected -- the reach radius (arm's
    # actual reach budget) must never grow past what the offset calibrated.
    # R7 T4: cup is exempt -- its stance is now VM B's proven fixed point
    # (config.CUP_GRASP_STANCE_XY), not the object-relative search this
    # invariant guards, and is asserted separately below.
    world = _NavTargetRecordingWorld(seed=3, head_placement="a")
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    plan_stage4(runner, world)

    assert world.nav_calls
    for obj_name, obj_xy, target_xy in world.nav_calls:
        if obj_name == "cup":
            assert target_xy == config.CUP_GRASP_STANCE_XY
            continue
        assert obj_xy is not None
        dist = math.hypot(target_xy[0] - obj_xy[0], target_xy[1] - obj_xy[1])
        assert dist <= STANCE_REACH_RADIUS_M + 1e-6


def test_nearest_sink_aim_point_matches_expected_travel_reduction():
    # T3 (LOOP_PROMPT_VM_A_REV4.md): every Stage-4 push used to aim at
    # SINK_CENTER_XY for every object, 30-40% farther than the scorer
    # requires. Expected numbers re-derived from SINK_BOUNDS/spawn XY,
    # matching the rev-4 audit's own table (handoff sec 105/106).
    #
    # REV14 merge note (2026-08-07): re-derived again after
    # SINK_AIM_MARGIN_M 0.05->0.12 (real scored Stage 4 point, commit
    # 8842dd7 on vm-b-explore) -- a larger margin pulls every aim point
    # farther in from SINK_BOUNDS, so travel distance grows for all four
    # objects. This test was stale on both pre-merge branches (failed on
    # vm-b-explore too, unmerged) -- the config value is the
    # GPU-verified fix; only this hardcoded literal needed updating.
    expected_travel_m = {
        "cup": 0.414,
        "spoon2": 0.531,
        "plate2": 0.534,
        "bowl2": 0.685,
    }
    spawn_xy = {
        "cup": (-4.1849, -1.7528),
        "spoon2": (-4.3415, -1.6781),
        "plate2": (-4.3087, -1.6608),
        "bowl2": (-4.2983, -1.4999),
    }
    for name, spawn in spawn_xy.items():
        aim = config.nearest_sink_aim_point(spawn)
        dist = math.hypot(aim[0] - spawn[0], aim[1] - spawn[1])
        assert abs(dist - expected_travel_m[name]) < 0.002, (name, dist, aim)
        # The aim point itself must always still score -- shrinking the
        # box by the margin must never push the target outside the real
        # (unshrunk) SINK_BOUNDS.
        assert config.scores_in_sink(aim[0], aim[1], config.SINK_TABLETOP_Z)


def test_nearest_sink_aim_point_stays_put_for_an_object_already_inside():
    # An object already inside the shrunk box should be aimed at its own
    # current position (zero extra travel), not dragged toward the centre.
    already_inside = (
        (config.SINK_BOUNDS["x_min"] + config.SINK_BOUNDS["x_max"]) / 2.0,
        (config.SINK_BOUNDS["y_min"] + config.SINK_BOUNDS["y_max"]) / 2.0,
    )
    aim = config.nearest_sink_aim_point(already_inside)
    assert (
        math.hypot(aim[0] - already_inside[0], aim[1] - already_inside[1])
        < 1e-9
    )


def test_plan_stage4_cleanup_aims_at_nearest_point_not_sink_center():
    # The cleanup call site (stages.py) must use the per-object aim point,
    # not the old fixed config.SINK_CENTER_XY, for every object. Capture
    # each object's XY at CALL TIME, before MockWorld's own carry_object_to
    # mutates it toward the target -- otherwise a naive post-hoc recompute
    # would compare against the object's post-move (already-in-sink) XY.
    world = _NavTargetRecordingWorld(seed=3, head_placement="a")
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    calls: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    real_carry = world.carry_object_to

    def _recording_carry(object_name, x, y, z=None, **p):
        pre_move_xy = world.object_xy(object_name)
        calls.append((object_name, (x, y), pre_move_xy))
        return real_carry(object_name, x, y, z, **p)

    world.carry_object_to = _recording_carry
    plan_stage4(runner, world)

    assert calls, "expected at least one carry_object_to call"
    for object_name, target_xy, pre_move_xy in calls:
        expected = config.nearest_sink_aim_point(pre_move_xy)
        assert target_xy == expected, (object_name, target_xy, expected)
        assert target_xy != config.SINK_CENTER_XY, (
            "aim point should be object-specific, not the fixed sink centre"
        )


def test_cleanup_retry_grid_real_ordering_covers_both_stances_within_budget():
    # T4 (LOOP_PROMPT_VM_A_REV4.md): confirm the REAL ordering
    # SelfCorrectingSkill/RetryPolicy actually produce, not an assumption.
    # "cleanup" is absent from outcomes.CLASSIFIERS so every failure here
    # classifies as UNSCORED -- _OUTCOME_PRIORITY_KEY has no entry for that
    # outcome, so diagnosis-driven reordering NEVER fires for this skill;
    # the plan is pure itertools.product(*grid.values()) declaration order.
    mem = ParamMemory()
    policy = RetryPolicy(mem)
    plan = policy.plan(
        "cleanup", head_placement="a", object_name="cup", last=None
    )

    assert len(plan) == policy.budget + 1 == 5

    # Every one of the first RETRY_BUDGET=4 real attempts must be pairwise
    # distinct on approach_stance -- the old grid order left this axis
    # fixed at "north" for all 4 (handoff sec 108's pasted real plan()
    # output before this fix).
    stances_seen = {p["approach_stance"] for p in plan[:4]}
    assert stances_seen == {"north", "east"}, plan[:4]

    # grasp_place requires a real hold and this project's own scored
    # command runs Stage 4 with --skip-grasp (2026-08-02 proof bundle's
    # repro command) -- under which it is a GUARANTEED no-op. It must
    # never occupy one of the scarce RETRY_BUDGET slots.
    assert all(p["method"] == "base_carry" for p in plan[:4]), plan[:4]

    # "controlled_slide" is a literal code-path duplicate of "base_carry"
    # (world_isaac.py carry_object_to's own comment) -- it must not appear
    # in the grid at all anymore, so no retry can be wasted re-running
    # identical code under a different label.
    all_methods = {v for p in plan for v in [p["method"]]}
    assert "controlled_slide" not in all_methods

    # The 4 real attempts must be pairwise distinct as whole parameter
    # sets (the bug this task fixes: previously method+approach_stance
    # were IDENTICAL across all 4, only the two offsets varied).
    as_tuples = [tuple(sorted(p.items())) for p in plan[:4]]
    assert len(set(as_tuples)) == 4, plan[:4]


def test_reach_gate_flag_default_on_and_disables_cleanly():
    # T5 (LOOP_PROMPT_VM_A_REV4.md): the reach-limit pre-flight gate must
    # default ON (unchanged behavior) and be fully bypassable via the new
    # flag for the A/B experiment.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod

    world_default = world_isaac_mod.IsaacWorld(simulation_app=None)
    assert world_default.reach_gate_enabled is True

    world_off = world_isaac_mod.IsaacWorld(
        simulation_app=None, reach_gate_enabled=False
    )
    assert world_off.reach_gate_enabled is False

    # _reach_limit_exceeded must short-circuit to False (never blocks)
    # when the gate is disabled, without even calling _arm_base_relative.
    def _boom(self, side, target):
        raise AssertionError(
            "_arm_base_relative should not be called when the gate is off"
        )

    world_off._arm_base_relative = types.MethodType(_boom, world_off)
    assert world_off._reach_limit_exceeded("right", (0.0, 0.0, 0.0)) is False


def test_already_scored_push_result_freezes_without_touching_object():
    # T3: a phase-boundary/retry-boundary guard must recognize an object
    # that already satisfies the scorer's own predicate and freeze there,
    # rather than let further (possibly destructive) push attempts touch
    # it again -- handoff sec 105's audit found the winning run's cup
    # survived THREE further failed attempts after it had already scored.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))

    scored_pos = (
        (config.SINK_BOUNDS["x_min"] + config.SINK_BOUNDS["x_max"]) / 2.0,
        (config.SINK_BOUNDS["y_min"] + config.SINK_BOUNDS["y_max"]) / 2.0,
        config.SINK_TABLETOP_Z + 0.01,
    )
    world.object_views = {"cup": _fake_object_view(scored_pos)}
    assert world._already_scored_push_result("cup") == {
        "scored": True,
        "already_scored": True,
    }

    world.object_views = {"cup": _fake_object_view((0.0, 0.0, 0.0))}
    assert world._already_scored_push_result("cup") is None


def test_push_object_to_refuses_to_touch_an_already_scoring_object():
    # world.arms is left at IsaacWorld's own __init__ default (None) -- if
    # the already-scored guard did NOT short-circuit before any contact/
    # motion code, this call would crash with AttributeError the first
    # time it touched self.arms (e.g. self.arms.release(...)), not just
    # silently redo the push. A guard that only prevented the SCORE from
    # changing but still drove the arm would pass a weaker assertion but
    # fail this one.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    scored_pos = (
        (config.SINK_BOUNDS["x_min"] + config.SINK_BOUNDS["x_max"]) / 2.0,
        (config.SINK_BOUNDS["y_min"] + config.SINK_BOUNDS["y_max"]) / 2.0,
        config.SINK_TABLETOP_Z + 0.01,
    )
    world.object_views = {"cup": _fake_object_view(scored_pos)}

    result = world._push_object_to(
        "right", "cup", *config.SINK_CENTER_XY, config.SINK_TABLETOP_Z
    )
    assert result == {"scored": True, "already_scored": True}


def test_stage_exception_is_isolated_and_chain_continues():
    # P0.2 (handoff.md §17.3): a world whose scoop() raises must not take the
    # episode down -- stages 3 and 4 must still be attempted and Stage 4's
    # frozen point must still be bankable in the same episode.
    class RaisingScoopWorld(MockWorld):
        def scoop(self, side, **p):
            raise AttributeError("scoop is not implemented on this world")

    world = RaisingScoopWorld(seed=7, head_placement="a")
    pipe = Task3Pipeline(world, memory_path=None)
    result = pipe.run_episode(seed=7, head_placement="a")

    assert result.aborted_at is None  # a raised exception is not a safety trip
    assert [s.stage for s in result.stages] == [1, 2, 3, 4]
    stage2 = result.stages[1]
    assert stage2.score == 0
    assert stage2.failure_reason is not None
    assert "AttributeError" in stage2.failure_reason
    # Stages 3 and 4 were genuinely attempted (not skipped) after the crash.
    assert result.stages[2].reports
    assert result.stages[3].reports
    assert result.highest_stage == 4


def test_stage_timeout_is_isolated_and_chain_continues():
    # Same contract, but for a stage that never returns (a real Isaac hang)
    # rather than one that raises. grasp() is used by both Stage 1 and Stage
    # 4, so both time out here -- the point is that Stage 2/3 (which don't
    # call grasp) still run in between, proving the hang didn't take the
    # whole episode down.
    class HangingGraspWorld(MockWorld):
        def grasp(self, side, object_name, **p):
            time.sleep(10)
            return super().grasp(side, object_name, **p)

    world = HangingGraspWorld(seed=7, head_placement="a")
    pipe = Task3Pipeline(world, memory_path=None, stage_wallclock_budget_s=0.2)
    result = pipe.run_episode(seed=7, head_placement="a")

    assert [s.stage for s in result.stages] == [1, 2, 3, 4]
    stage1, stage2, _stage3, stage4 = result.stages
    assert stage1.score == 0
    assert stage1.failure_reason is not None
    assert "TimeoutError" in stage1.failure_reason
    assert stage4.failure_reason is not None
    assert "TimeoutError" in stage4.failure_reason
    # Stage 2 (no grasp() call) was genuinely attempted between the two
    # timed-out stages, not skipped.
    assert stage2.reports
    assert stage2.failure_reason is None


# ---- P0.9: development_score / official_spec_ready split ---------------- #


def test_official_spec_ready_stage1_always_false() -> None:
    # grading.score_stage1_table_setup never reads assigned-seat identity
    # (handoff sec 16.2 row 7) -- a dev-scorer pass is never a real-rules
    # pass yet, regardless of details.
    assert official_spec_ready(1, {"passed": ["cup"]}) is False
    assert official_spec_ready(1, {}) is False


def test_official_spec_ready_stage2_requires_safe_measured_force() -> None:
    assert official_spec_ready(2, {"peak_head_force_n": 12.0}) is True
    assert (
        official_spec_ready(2, {"peak_head_force_n": 65.0}) is True
    )  # inclusive
    assert official_spec_ready(2, {"peak_head_force_n": 65.1}) is False
    assert official_spec_ready(2, {}) is False  # sensor never wired
    assert official_spec_ready(2, {"peak_head_force_n": None}) is False


def test_official_spec_ready_stage4_true() -> None:
    assert official_spec_ready(4, {}) is True


def test_episode_json_carries_development_score_and_official_spec_ready():
    world = MockWorld(seed=42, head_placement="a")
    pipe = Task3Pipeline(world, memory_path=None)
    result = pipe.run_episode(seed=42, head_placement="a")
    import json

    payload = json.loads(result.as_json())
    for stage_entry in payload["per_stage"]:
        assert stage_entry["development_score"] == stage_entry["score"]
        assert isinstance(stage_entry["official_spec_ready"], bool)
    stage1_entry = payload["per_stage"][0]
    assert stage1_entry["official_spec_ready"] is False  # sec 16.2 row 7
    stage2_entry = payload["per_stage"][1]
    # MockWorld never measures a real head force -- must never claim ready.
    assert stage2_entry["official_spec_ready"] is False


def test_timed_out_stage_thread_fully_joined_before_next_stage_starts():
    # handoff sec 4.64: a REAL GPU run showed a timed-out stage's thread
    # was left running while the next stage's thread started against the
    # same (non-thread-safe) world -- two threads' commands raced and
    # nothing ever converged. Prove the fix with a concurrency-detecting
    # lock, not timing: no two stage plans may ever be "active" at once,
    # even when one overruns its budget.
    import threading
    import time

    import task3_pipeline.orchestrator as orch_mod

    active = {"count": 0, "max_concurrent": 0}
    lock = threading.Lock()

    def _mark_active():
        with lock:
            active["count"] += 1
            active["max_concurrent"] = max(
                active["max_concurrent"], active["count"]
            )

    def _mark_inactive():
        with lock:
            active["count"] -= 1

    def slow_plan(runner, world):
        _mark_active()
        try:
            time.sleep(0.3)  # longer than the 0.05s budget below
        finally:
            _mark_inactive()
        return StageResult(1, 0, 4, [], {})

    def fast_plan(runner, world):
        _mark_active()
        try:
            pass
        finally:
            _mark_inactive()
        return StageResult(2, 0, 4, [], {})

    original_plans = dict(orch_mod.STAGE_PLANS)
    orch_mod.STAGE_PLANS[1] = slow_plan
    orch_mod.STAGE_PLANS[2] = fast_plan
    try:
        world = MockWorld(seed=1, head_placement="a")
        pipe = Task3Pipeline(
            world, memory_path=None, stage_wallclock_budget_s=0.05
        )
        result = pipe.run_episode(seed=1, head_placement="a", order=(1, 2))
    finally:
        orch_mod.STAGE_PLANS.clear()
        orch_mod.STAGE_PLANS.update(original_plans)

    assert active["max_concurrent"] == 1, (
        "two stage plans were active against the same world at once"
    )
    assert result.stages[0].failure_reason is not None
    assert "TimeoutError" in result.stages[0].failure_reason
    assert result.stages[1].failure_reason is None


def test_worker_dying_without_reporting_does_not_hang_the_episode():
    # sec 19b W1.4: _target() only catches `Exception`, so a worker thread
    # that dies from a BaseException (e.g. SystemExit) joins successfully
    # (worker.is_alive() is False) but never calls outcome.put() -- a bare
    # outcome.get() would then block the main thread forever with no
    # diagnostic. A stage plan that raises SystemExit reproduces exactly
    # that: the thread dies before put() ever runs.
    import task3_pipeline.orchestrator as orch_mod

    def dying_plan(runner, world):
        raise SystemExit("simulated worker death")

    def fast_plan(runner, world):
        return StageResult(2, 0, 4, [], {})

    original_plans = dict(orch_mod.STAGE_PLANS)
    orch_mod.STAGE_PLANS[1] = dying_plan
    orch_mod.STAGE_PLANS[2] = fast_plan
    try:
        world = MockWorld(seed=1, head_placement="a")
        pipe = Task3Pipeline(
            world, memory_path=None, stage_wallclock_budget_s=5.0
        )
        result = pipe.run_episode(seed=1, head_placement="a", order=(1, 2))
    finally:
        orch_mod.STAGE_PLANS.clear()
        orch_mod.STAGE_PLANS.update(original_plans)

    assert result.stages[0].score == 0
    assert result.stages[0].failure_reason is not None
    assert "WorkerDiedError" in result.stages[0].failure_reason
    # Stage 2 still ran -- the main thread did not hang on stage 1's queue.
    assert result.stages[1].failure_reason is None


def test_matrix_majority_pass():
    world = MockWorld()
    pipe = Task3Pipeline(world, memory_path=None)
    pcts = []
    for hp in "abcdefghi":
        for seed in range(5):
            pcts.append(pipe.run_episode(seed=seed, head_placement=hp).pct)
    passed = sum(1 for p in pcts if p >= 0.70)
    assert passed / len(pcts) >= 0.70


def test_app_launcher_config_gates_cameras_on_record_video():
    # W0.1 (handoff sec 18.2): enable_cameras used to be hard-coded True on
    # every real run. Pure-function check so this never needs isaaclab
    # installed to verify.
    from task3_pipeline.run_task3 import _app_launcher_config, build_parser

    args_off = build_parser().parse_args(["--seed", "7"])
    assert _app_launcher_config(args_off)["enable_cameras"] is False

    args_on = build_parser().parse_args(["--seed", "7", "--record-video"])
    assert _app_launcher_config(args_on)["enable_cameras"] is True


def test_navigate_to_second_call_not_pinned_by_stale_base_hold_anchor():
    # sec 21 Bug A (owner, confirmed live on a Lightning GPU 2026-07-26):
    # _tick() unconditionally re-issues a hold-toward-anchor twist whenever
    # self._base_hold_anchor is set, and navigate_to() sets that anchor at
    # the END of every call (so a later manipulation phase can hold
    # position) but never clears it at the START of its own loop. A SECOND
    # navigate_to() call (second object, or a retry) therefore has its own
    # commanded twist overwritten every tick by base_twist_toward(the FIRST
    # call's endpoint) -- near-zero error at the start of the second call,
    # so the base freezes there instead of driving to the new target.
    # carry_object_to() already guards against exactly this
    # (`self._base_hold_anchor = None` before its own loop, world_isaac.py);
    # navigate_to() did not. This test exercises the real navigate_to()/
    # _tick() production code with only the adapter/sim/nav-skill faked.
    import math
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    class _StubNav:
        def __init__(self, target_xy, target_yaw=None, **kw):
            self.target_xy = target_xy

        def compute(self, pose):
            tx, ty = self.target_xy
            dx, dy = tx - pose.x, ty - pose.y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                return 0.0, 0.0, True
            speed = 0.5
            return speed * dx / dist, speed * dy / dist, False

    class _FakeAdapter:
        def __init__(self):
            self.x = self.y = self.yaw = 0.0
            self._vx = self._vy = 0.0

        def pose(self):
            return Pose2D(self.x, self.y, self.yaw)

        def apply_twist(self, vx, vy, hold_heading=False):
            self._vx, self._vy = vx, vy

        def integrate(self, dt):
            self.x += self._vx * dt
            self.y += self._vy * dt

    class _FakeSim:
        def __init__(self, adapter, dt=0.005):
            self._adapter = adapter
            self.cfg = types.SimpleNamespace(dt=dt)

        def step(self, render=None):
            self._adapter.integrate(self.cfg.dt)

    class _FakeScene:
        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    adapter = _FakeAdapter()
    world.adapter = adapter
    world.sim = _FakeSim(adapter)
    world.scene = _FakeScene()
    world.robot = None
    world.record_video = False
    world.arms = _FakeArmsForGraspFrame((0.0, 0.0, 0.0))
    world._m = {
        "disable_robot_external_wrenches": lambda robot: None,
        "NavigateTo": _StubNav,
        # Q1 merge (vm-a-control x vm-b-explore): navigate_to() now also
        # tucks the arm via self._m["ramp_arm_pose"] before its own base
        # motion (M1, world_isaac.py ~588) -- a call this stub predates.
        # No-op stand-ins: this test only exercises the base-hold-anchor
        # bug, not the arm ramp itself (that has its own real-code test in
        # task3_autonomy/tests/test_skills.py).
        "ramp_arm_pose": lambda robot, pose, **kw: None,
        "TRANSIT_ARM_POSE": {},
    }

    first = world.navigate_to(1.0, 0.0, budget_s=10.0)
    assert first["terminal_error_m"] < 0.1

    second = world.navigate_to(3.0, 0.0, budget_s=10.0)
    assert second["terminal_error_m"] < 0.5, (
        "base got pinned by a stale _base_hold_anchor from the first "
        f"call instead of reaching the second target: {second}"
    )


def test_rotate_to_not_damped_by_stale_base_hold_anchor():
    # sec 21 Bug A's sibling (owner-diagnosed live on a Lightning GPU
    # 2026-07-26, right after the same session's Bug A fix landed):
    # _rotate_to() has the identical missing-anchor-clear defect as
    # navigate_to() did. TmrBaseAdapter.apply_twist(vx, vy, wz_cmd=0.0, *,
    # hold_heading=False) (scripts/common/tmr_base_control.py's
    # compensate_yaw_rate) treats a call with wz_cmd==0 and
    # hold_heading=True as "hold whatever heading was last locked in" --
    # it ADDS a KP/KD correction on top of wz_cmd rather than passing
    # nothing through. Since `_tick()`'s hold-anchor block calls
    # apply_twist(hold_vx, hold_vy, hold_heading=True) (wz_cmd defaults to
    # 0.0) immediately after `_rotate_to()`'s own
    # apply_twist(0, 0, wz_from_skill) every tick, and the LAST call before
    # sim.step() is what the wheels actually receive, the hold call's
    # damping-only wz overrides `_rotate_to()`'s intended rotation on every
    # tick whenever a stale `_base_hold_anchor` is set (e.g. immediately
    # after a preceding navigate_to() call, which always sets one at the
    # end). Confirmed live: a real `_rotate_to` call after a successful
    # navigate_to() oscillated (yaw bouncing between -1.74 and -2.06 rad)
    # instead of converging to a nearby target, then hit the W0.4 watchdog.
    #
    # This fake adapter reimplements compensate_yaw_rate's exact branching
    # (manual_rotation / hold_while_stopped / KP-KD-on-yaw-rate) so the test
    # exercises the real interaction, not a simplified stand-in.
    import math

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D, wrap_to_pi
    from task3_autonomy.skills import RotateTo

    class _FakeAdapter:
        STOP_EPS = 0.01
        HEADING_HOLD_KP = 2.0
        HEADING_HOLD_KD = 0.1
        MAX_HEADING_COMP_RADPS = 1.0

        def __init__(self, yaw=0.0):
            self.x = self.y = 0.0
            self.yaw = yaw
            self._hold_yaw = yaw
            self._yaw_rate = 0.0
            self._vx = self._vy = self._wz = 0.0

        def pose(self):
            return Pose2D(self.x, self.y, self.yaw)

        def apply_twist(self, vx, vy, wz_cmd=0.0, *, hold_heading=False):
            manual_rotation = abs(wz_cmd) > 1.0e-4
            stopped = math.hypot(vx, vy) < self.STOP_EPS
            if manual_rotation or (stopped and not hold_heading):
                wz = wz_cmd
                self._hold_yaw = self.yaw
            else:
                yaw_error = wrap_to_pi(self._hold_yaw - self.yaw)
                compensation = (
                    self.HEADING_HOLD_KP * yaw_error
                    - self.HEADING_HOLD_KD * self._yaw_rate
                )
                compensation = max(
                    -self.MAX_HEADING_COMP_RADPS,
                    min(self.MAX_HEADING_COMP_RADPS, compensation),
                )
                wz = wz_cmd + compensation
            self._vx, self._vy, self._wz = vx, vy, wz

        def integrate(self, dt):
            self.x += self._vx * dt
            self.y += self._vy * dt
            self.yaw = wrap_to_pi(self.yaw + self._wz * dt)
            self._yaw_rate = self._wz

    class _FakeSim:
        def __init__(self, adapter, dt=0.005):
            import types

            self._adapter = adapter
            self.cfg = types.SimpleNamespace(dt=dt)

        def step(self, render=None):
            self._adapter.integrate(self.cfg.dt)

    class _FakeScene:
        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    adapter = _FakeAdapter(yaw=0.0)
    world.adapter = adapter
    world.sim = _FakeSim(adapter)
    world.scene = _FakeScene()
    world.robot = None
    world.record_video = False
    world.arms = None
    world._m = {
        "disable_robot_external_wrenches": lambda robot: None,
        "RotateTo": RotateTo,
    }
    # A stale anchor at the base's current position, exactly what
    # navigate_to() leaves behind on success.
    world._base_hold_anchor = (0.0, 0.0)

    ok = world._rotate_to(math.pi / 2, budget_s=10.0)

    assert ok, (
        f"_rotate_to did not converge with a stale _base_hold_anchor set "
        f"(final yaw={adapter.yaw:.3f}, target={math.pi / 2:.3f})"
    )


def test_run_task3_skip_navigation_flag_reaches_isaac_world(monkeypatch):
    # W0.3 (handoff sec 18.3): --skip-navigation must reach IsaacWorld's
    # constructor. world_isaac.py has zero Isaac imports at module scope
    # (its own docstring), so it can be imported and its IsaacWorld symbol
    # stubbed out here without isaaclab installed.
    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_pipeline.run_task3 import _make_world, build_parser

    captured: dict = {}

    class _StubIsaacWorld:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(world_isaac_mod, "IsaacWorld", _StubIsaacWorld)

    args = build_parser().parse_args(
        ["--seed", "7", "--head-placement", "a", "--skip-navigation"]
    )
    _make_world(args, simulation_app=object())
    assert captured["skip_navigation"] is True


def test_run_task3_skip_navigation_defaults_false(monkeypatch):
    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_pipeline.run_task3 import _make_world, build_parser

    captured: dict = {}

    class _StubIsaacWorld:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(world_isaac_mod, "IsaacWorld", _StubIsaacWorld)

    args = build_parser().parse_args(["--seed", "7", "--head-placement", "a"])
    _make_world(args, simulation_app=object())
    assert captured["skip_navigation"] is False


def test_mock_world_unaffected_by_skip_navigation_flag():
    # --mock must ignore --skip-navigation entirely (it has no concept of
    # a spawn stance) -- gate from handoff sec 18.6 W0.3.
    from task3_pipeline.run_task3 import _make_world, build_parser

    args = build_parser().parse_args(
        ["--mock", "--seed", "7", "--head-placement", "a", "--skip-navigation"]
    )
    world = _make_world(args)
    assert isinstance(world, MockWorld)


# ---- W0.2: tick_count / wall_time_seconds / s_per_tick ------------------- #


def test_episode_json_carries_tick_telemetry_for_mock():
    # handoff sec 18.1c/18.6 W0.2: every result JSON must carry these three
    # fields. MockWorld has no simulation ticks, so tick_count is 0 and
    # s_per_tick must be None (never a fabricated rate), while
    # wall_time_seconds is always a real measured wall-clock duration.
    import json

    world = MockWorld(seed=42, head_placement="a")
    pipe = Task3Pipeline(world, memory_path=None)
    result = pipe.run_episode(seed=42, head_placement="a")
    payload = json.loads(result.as_json())

    assert payload["tick_count"] == 0
    assert payload["s_per_tick"] is None
    assert payload["wall_time_seconds"] >= 0.0
    assert result.tick_count == 0
    assert result.s_per_tick is None


def test_episode_result_s_per_tick_derives_from_tick_count():
    # A world reporting real ticks (as IsaacWorld does via self._tick_count)
    # must produce a real s_per_tick, not None.
    class TickingWorld(MockWorld):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._tick_count = 0

        def navigate_to(self, x, y, yaw=None, **p):
            self._tick_count += 100
            return super().navigate_to(x, y, yaw, **p)

    world = TickingWorld(seed=1, head_placement="a")
    pipe = Task3Pipeline(world, memory_path=None)
    result = pipe.run_episode(seed=1, head_placement="a")

    assert result.tick_count > 0
    assert result.s_per_tick is not None
    assert result.s_per_tick == round(
        result.wall_time_seconds / result.tick_count, 4
    )


# ---- W0.5: tick-denominated stage budgets -------------------------------- #


def test_stage_tick_budgets_defined_for_all_four_stages():
    for stage in (1, 2, 3, 4):
        assert config.STAGE_TICK_BUDGETS[stage] > 0


def test_stage_wallclock_ceiling_derives_from_tick_budget_and_rate():
    # rate chosen so the derived value stays under HARD_JOIN_CEILING_S --
    # this test is about the derivation arithmetic, not the cap.
    ceiling = config.stage_wallclock_ceiling_s(1, measured_s_per_tick=0.001)
    assert ceiling == config.STAGE_TICK_BUDGETS[1] * 0.001 * 1.5
    assert ceiling < config.HARD_JOIN_CEILING_S


def test_stage_wallclock_ceiling_falls_back_when_rate_not_measured():
    ceiling = config.stage_wallclock_ceiling_s(1)
    # sec 19b: stage 1's fallback-derived ceiling (79,972s = 22.2h) is
    # exactly the silent-wall case HARD_JOIN_CEILING_S exists to prevent,
    # so the capped value -- not the raw derivation -- is correct here.
    assert ceiling == config.HARD_JOIN_CEILING_S


def test_stage_wallclock_ceiling_never_exceeds_hard_cap():
    for stage in (1, 2, 3, 4):
        assert (
            config.stage_wallclock_ceiling_s(stage)
            <= config.HARD_JOIN_CEILING_S
        )


def test_pipeline_default_budget_is_derived_not_flat_600():
    # W0.5: the old flat STAGE_WALLCLOCK_BUDGET_S=600 default must no
    # longer be what a fresh Task3Pipeline uses.
    world = MockWorld(seed=1, head_placement="a")
    pipe = Task3Pipeline(world, memory_path=None)
    assert pipe.stage_wallclock_budget_s is None
    assert pipe.measured_s_per_tick is None


def test_official_spec_ready_stage3_forced_false():
    # CORRECTION (handoff sec 18.4): stage 3 used to hard-return True.
    # BEAN_RECOVERY_CENTER sits on the beans' own start XY -- forced False
    # until the organizer prose is re-read.
    assert official_spec_ready(3, {}) is False
    assert official_spec_ready(3, {"ratio": 1.0}) is False


def test_run_task3_cli_order_flag_isolates_a_single_stage(capsys):
    # handoff sec 4.60: lets a single stage be exercised on real Isaac
    # without Stage 1's slow per-object loop. --mock coverage only here;
    # the real-Isaac wiring is the same _make_world()/run_episode() path
    # already covered by the full-chain tests above.
    import json

    from task3_pipeline.run_task3 import build_parser, run_one

    args = build_parser().parse_args(
        ["--mock", "--seed", "7", "--head-placement", "a", "--order", "2"]
    )
    run_one(args)
    payload = json.loads(capsys.readouterr().out)
    assert [s["stage"] for s in payload["per_stage"]] == [2]
    assert payload["highest_stage_completed"] in (0, 2)


class _FakePositions:
    """Stands in for the tensor-like object IsaacWorld.object_position()
    expects (it calls ``.tolist()`` on whatever get_world_poses() returns)."""

    def __init__(self, position):
        self._position = list(position)

    def tolist(self):
        return [self._position]


def _fake_object_view(position):
    class _View:
        def get_world_poses(self):
            return _FakePositions(position), None

    return _View()


class _FakeArmsForGraspFrame:
    """Minimal arms stub: only the calls grasp()/_log_phase() make."""

    spine = 0.0
    _default_gripper_effort_limits = {"left": 8.0, "right": 8.0}

    def __init__(self, ee_position, holding=True, gripper_rad=0.076):
        self._ee_position = ee_position
        self._holding = holding
        self._gripper_rad = gripper_rad

    def grasp(self, side, **kw):
        return self._holding

    def gripper_position(self, side):
        return self._gripper_rad

    def commanded_arm_joint_positions(self, side):
        # `_push_object_to`'s gentle ramp applies the same joint-space
        # delta guard `reach()` uses. A constant vector means "no thrash",
        # which is what this stub should represent.
        return [0.0] * 7

    def measured_arm_joints(self, side):
        return [0.0] * 7

    def _gripper_position_upper_limit(self, side):
        # The real asset's authored open limit, measured directly
        # (scripts/task3/probe_gripper_joint_limits.py): both sides
        # [0.0, 0.8203047513961792]. grasp()'s honest_hold reads this
        # rather than a module constant so the two cannot drift apart.
        return 0.8203047513961792

    def ee_world_poses(self):
        pose = (self._ee_position, (1.0, 0.0, 0.0, 0.0))
        return [pose, pose]

    def measured_spine_position(self):
        return 0.0

    def sync_targets_from_measured(self):
        pass


def test_grasp_honest_hold_uses_grasp_frame_not_raw_object_distance():
    # ACTIVE_BRIEF.md sec 3/5, M0: world_isaac.py's grasp() used to compare
    # the wrist against the object's raw origin, which always includes the
    # commanded standoff (0.068-0.10 m along the tool axis) -- a perfect
    # grasp could never pass THRESHOLDS.GRASP_HELD_MAX_DIST_M (0.08 m). The
    # fix measures from the grasp frame instead: the object's current
    # position shifted by the same offset recorded when the descend target
    # was computed. This exercises the real production grasp() with only
    # the arms/object-view/vgl seams faked.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world._m = {
        "vgl": type(
            "_Vgl",
            (),
            {
                "object_follows_end_effector": staticmethod(
                    lambda op, ep, max_distance_m: math.dist(op, ep)
                    <= max_distance_m
                )
            },
        )()
    }

    object_pos = (1.0, 2.0, 0.5)
    standoff = (0.0, 0.0, 0.09)  # exceeds GRASP_HELD_MAX_DIST_M (0.08) alone
    world.object_views = {"plate2": _fake_object_view(object_pos)}
    world._last_grasp_offset = {"plate2": standoff}

    # Case 1: EE landed exactly on the grasp frame (object + standoff) --
    # a real, honest grasp. The OLD code compared raw object_pos to this
    # same ee_pos and would have measured 0.09 m > 0.08 m and called it a
    # miss even though the grasp is perfect.
    ee_at_grasp_frame = (
        object_pos[0] + standoff[0],
        object_pos[1] + standoff[1],
        object_pos[2] + standoff[2],
    )
    world.arms = _FakeArmsForGraspFrame(ee_at_grasp_frame)
    result = world.grasp("right", "plate2")
    assert result["object_follows_ee"] is True, result
    assert result["object_ee_dist_m"] < 0.01, result
    assert (
        result["object_ee_dist_raw_m"]
        > config.THRESHOLDS.GRASP_HELD_MAX_DIST_M
    ), (
        "raw distance should still show the standoff, proving the OLD gate "
        f"would have failed a real grasp: {result}"
    )

    # Case 2: the object is nowhere near the grasp frame (a real miss --
    # gripper closed on empty air). Must still be rejected.
    world.object_views = {"plate2": _fake_object_view((1.3, 2.0, 0.5))}
    world.arms = _FakeArmsForGraspFrame(ee_at_grasp_frame)
    result = world.grasp("right", "plate2")
    assert result["object_follows_ee"] is False, result


def test_grasp_contact_reflects_real_holding_not_hardcoded_true():
    """REV12 follow-up (T7 finding, plans/SYNC.md 2026-08-06): grasp()'s
    returned "contact" field used to be a hardcoded True literal
    regardless of `holding` (the real signal from arms.grasp()) -- this
    silently defeated outcomes.classify_grasp's `contacted` gate. Real
    T7 episodes 103/104 hit exactly this: the gripper never actually
    closed (holding=False) but object_follows_ee happened to be True
    (the object was within distance tolerance anyway), and nothing in
    the returned dict could tell a caller the gripper hadn't closed."""
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D
    from task3_pipeline.outcomes import SkillOutcome, classify_grasp

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world._m = {
        "vgl": type(
            "_Vgl",
            (),
            {
                "object_follows_end_effector": staticmethod(
                    lambda op, ep, max_distance_m: math.dist(op, ep)
                    <= max_distance_m
                )
            },
        )()
    }

    object_pos = (1.0, 2.0, 0.5)
    world.object_views = {"bowl2": _fake_object_view(object_pos)}
    world._last_grasp_offset = {}
    world._pre_lift_baseline = {}

    # holding=False (gripper never actually closed/converged) but the EE
    # happens to sit right at the object -- follows_ee would read True
    # from distance alone.
    world.arms = _FakeArmsForGraspFrame(object_pos, holding=False)
    result = world.grasp("right", "bowl2")

    assert result["object_follows_ee"] is True, (
        "test setup: distance check must pass despite holding=False"
    )
    assert result["contact"] is False, (
        f"contact must reflect real holding=False, not a hardcoded "
        f"True: {result}"
    )
    assert world._held is None, (
        "self._held must stay None when holding was False, even though "
        f"object_follows_ee looked satisfied: {result}"
    )
    # The real point of the fix: classify_grasp must now reach MISS
    # instead of silently reporting SUCCESS/WEAK_GRASP off a fake
    # "contact": True.
    outcome, _diag = classify_grasp(result)
    assert outcome is SkillOutcome.MISS, (
        f"expected MISS with contact=False, got {outcome}: {result}"
    )


class _FakeArmsWithCloseTelemetry(_FakeArmsForGraspFrame):
    """REV13 T2: like `_FakeArmsForGraspFrame`, but actually populates the
    `telemetry` dict `world_isaac.py::grasp()` now passes through to
    `arms.grasp()`, the way the real `DualArmController.grasp()` does via
    `run_gripper_close_ramp`. Proves the plumbing between the two files,
    not just the pure tick-loop function in isolation."""

    def grasp(self, side, **kw):
        telemetry = kw.get("telemetry")
        if telemetry is not None:
            telemetry["ticks"] = [
                {
                    "tick": 1,
                    "commanded_target_rad": 0.45,
                    "measured_position_rad": self._gripper_rad,
                    "error_rad": round(0.45 - self._gripper_rad, 5),
                }
            ]
            telemetry["tick_count"] = 1
            telemetry["contact_tick"] = 1
            telemetry["final_residual_rad"] = self._gripper_rad
            telemetry["outcome"] = "contact_sustained"
            telemetry["holding"] = self._holding
        return self._holding


def test_world_grasp_returns_close_telemetry_from_arms():
    """GATE T2: the new telemetry fields must actually populate through
    `world_isaac.py::grasp()`'s real call to `self.arms.grasp(...,
    telemetry=...)`, not just in the pure `run_gripper_close_ramp` unit
    tests (`task3_autonomy/tests/test_arms.py`)."""
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world._m = {
        "vgl": type(
            "_Vgl",
            (),
            {
                "object_follows_end_effector": staticmethod(
                    lambda op, ep, max_distance_m: math.dist(op, ep)
                    <= max_distance_m
                )
            },
        )()
    }

    object_pos = (1.0, 2.0, 0.5)
    world.object_views = {"bowl2": _fake_object_view(object_pos)}
    world._last_grasp_offset = {}
    world._pre_lift_baseline = {}
    world.arms = _FakeArmsWithCloseTelemetry(
        object_pos, holding=True, gripper_rad=0.2979
    )

    result = world.grasp("right", "bowl2")

    assert "close_telemetry" in result, result
    telemetry = result["close_telemetry"]
    assert telemetry["tick_count"] == 1
    assert telemetry["contact_tick"] == 1
    assert telemetry["outcome"] == "contact_sustained"
    assert math.isclose(telemetry["final_residual_rad"], 0.2979)
    assert math.isclose(telemetry["ticks"][0]["measured_position_rad"], 0.2979)


def test_world_grasp_returns_authored_effort_limit():
    # REV14 T4 hypothesis 1: the authored gripper effort limit has never
    # been logged anywhere in this project's history. world.grasp() must
    # surface it (from arms._default_gripper_effort_limits[side]) so a
    # real GPU episode's log can be checked for a suspiciously low
    # ceiling without needing a new Isaac Lab API.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world._m = {
        "vgl": type(
            "_Vgl",
            (),
            {
                "object_follows_end_effector": staticmethod(
                    lambda op, ep, max_distance_m: math.dist(op, ep)
                    <= max_distance_m
                )
            },
        )()
    }

    object_pos = (1.0, 2.0, 0.5)
    world.object_views = {"bowl2": _fake_object_view(object_pos)}
    world._last_grasp_offset = {}
    world._pre_lift_baseline = {}
    world.arms = _FakeArmsWithCloseTelemetry(
        object_pos, holding=True, gripper_rad=0.2979
    )
    world.arms._default_gripper_effort_limits = {
        "left": 8.0,
        "right": 12.5,
    }

    result = world.grasp("right", "bowl2")

    assert math.isclose(result["authored_effort_limit_nm"], 12.5)


class _SequencedObjectView:
    """Like _fake_object_view, but returns a new z each call -- lets a
    test drive world.lift()'s z_before/per-tick/z_after reads through a
    scripted trajectory instead of one fixed position."""

    def __init__(self, xy, z_sequence):
        self._xy = xy
        self._z_sequence = list(z_sequence)
        self._calls = 0

    def get_world_poses(self):
        idx = min(self._calls, len(self._z_sequence) - 1)
        z = self._z_sequence[idx]
        self._calls += 1
        return _FakePositions((self._xy[0], self._xy[1], z)), None


class _FakeArmsForLiftTelemetry(_FakeArmsForGraspFrame):
    """Drives world.lift()'s on_tick callback with a fixed number of
    ticks, mirroring _FakeArmsWithCloseTelemetry's role for grasp()."""

    def __init__(self, ee_position, tick_count):
        super().__init__(ee_position)
        self._tick_count = tick_count

    def lift(self, side, dz, **kw):
        on_tick = kw.get("on_tick")
        if on_tick is not None:
            for tick in range(self._tick_count):
                on_tick(tick)
        return True


def test_world_lift_returns_per_tick_telemetry_and_first_slip_tick():
    """REV13 T4-followup-2 (plans/SYNC.md 2026-08-07): 9/9 real episodes
    with a telemetrically "sustained" close still failed to lift -- the
    close loop was never the bottleneck. This proves world.lift()'s new
    on_tick wiring actually samples object-vs-EE tracking during the
    lift and reports the first tick it stopped following, the way T2's
    close_telemetry reports contact_tick, not just an aggregate rise."""
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world._m = {
        "vgl": type(
            "_Vgl",
            (),
            {
                "object_follows_end_effector": staticmethod(
                    lambda op, ep, max_distance_m: math.dist(op, ep)
                    <= max_distance_m
                )
            },
        )()
    }

    ee_position = (1.0, 2.0, 0.9)
    # Object starts at the grasp frame (follows), then slips away at
    # tick 2 (its xy stays under the EE but z drifts far below it), then
    # a final read for z_after.
    world.object_views = {
        "bowl2": _SequencedObjectView(
            (1.0, 2.0), [0.9, 0.9, 0.9, 0.3, 0.3, 0.3]
        )
    }
    world._last_grasp_offset = {}
    world._pre_lift_baseline = {}
    world._held = "bowl2"
    world.arms = _FakeArmsForLiftTelemetry(ee_position, tick_count=4)

    result = world.lift("right", 0.3)

    assert "lift_telemetry" in result, result
    telemetry = result["lift_telemetry"]
    assert [row["tick"] for row in telemetry] == [0, 1, 2, 3]
    assert [row["object_follows_ee"] for row in telemetry] == [
        True,
        True,
        False,
        False,
    ], telemetry
    assert result["lift_first_slip_tick"] == 2, result


class _FakeBodyRow:
    """A single tensor row: supports the real Isaac tensor's .tolist()."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeBodyTensor:
    """Stands in for IsaacWorld._arm_base_relative()'s
    self.robot.data.body_pos_w/body_quat_w -- supports the same
    tensor[0, idx].tolist() access pattern real Isaac tensors do."""

    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        _, idx = key
        return _FakeBodyRow(self._rows[idx])


def _fake_robot(body_names, positions, quats):
    import types

    return types.SimpleNamespace(
        body_names=body_names,
        data=types.SimpleNamespace(
            body_pos_w=_FakeBodyTensor(positions),
            body_quat_w=_FakeBodyTensor(quats),
        ),
    )


# R9 T3 (plans/LOOP_PROMPT_VM_A_REV9.md): _select_arm_side() now tries a
# real IK feasibility query (_ik_feasible_sides) BEFORE falling back to
# distance -- but that query needs `world.arms` (built by reset(), never
# called in these three tests), so `_ik_feasible_sides` returns None here
# and every one of the next three tests exercises exactly the same
# distance-only fallback path as before the change, unmodified on
# purpose. The new feasibility-driven behavior gets its own tests below
# (test_select_arm_side_prefers_ik_feasible_side_over_distance and
# test_select_arm_side_bowl2_case_selects_left).
def test_select_arm_side_picks_the_geometrically_nearer_arm():
    # P5 (plans/LOOP_PROMPT_VM_A.md rev 2): _select_arm_side() must use the
    # SAME _arm_base_relative() the telemetry already reports, not a
    # reimplementation -- this exercises the real production method.
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.robot = _fake_robot(
        body_names=["root", "left_base", "right_base"],
        positions=[[0.0, 0.0, 0.0], [-1.0, 0.05, 0.5], [-1.0, -0.05, 0.5]],
        quats=[[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    # Object sits closer to left_base (y=0.05) than right_base (y=-0.05).
    world.object_views = {"bowl2": _fake_object_view((-1.0, 0.5, 0.5))}

    assert world._select_arm_side("bowl2") == "left"


def test_select_arm_side_falls_back_to_right_when_side_is_further():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.robot = _fake_robot(
        body_names=["root", "left_base", "right_base"],
        positions=[[0.0, 0.0, 0.0], [-1.0, 0.05, 0.5], [-1.0, -0.05, 0.5]],
        quats=[[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    # Object sits closer to right_base this time.
    world.object_views = {"bowl2": _fake_object_view((-1.0, -0.5, 0.5))}

    assert world._select_arm_side("bowl2") == "right"


def test_select_arm_side_defaults_to_right_when_body_lookup_misses():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.robot = _fake_robot(
        body_names=["root"],  # no left_base/right_base at all
        positions=[[0.0, 0.0, 0.0]],
        quats=[[1.0, 0.0, 0.0, 0.0]],
    )
    world.object_views = {"bowl2": _fake_object_view((-1.0, 0.5, 0.5))}

    assert world._select_arm_side("bowl2") == "right"


class _FakeIkForArmSide:
    def __init__(self, left_succeeded: bool, right_succeeded: bool):
        self._left_succeeded = left_succeeded
        self._right_succeeded = right_succeeded

    def solve(self, *args, **kwargs):
        import types

        return types.SimpleNamespace(
            left_succeeded=self._left_succeeded,
            right_succeeded=self._right_succeeded,
        )


class _FakeArmsForArmSide:
    """Minimal `self.arms` stub for `_ik_feasible_sides`: only `spine`,
    `_root_pose`, and `_ik.solve` -- the exact surface it calls."""

    def __init__(self, left_succeeded: bool, right_succeeded: bool):
        self.spine = 0.4
        self._ik = _FakeIkForArmSide(left_succeeded, right_succeeded)

    def _root_pose(self, robot):
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)


def test_select_arm_side_prefers_ik_feasible_side_over_distance():
    # Same geometry as test_select_arm_side_falls_back_to_right_when_side_
    # is_further (object closer to right_base) -- but the feasibility
    # query says the opposite. R9 T3's whole point: feasibility must win.
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.robot = _fake_robot(
        body_names=["root", "left_base", "right_base"],
        positions=[[0.0, 0.0, 0.0], [-1.0, 0.05, 0.5], [-1.0, -0.05, 0.5]],
        quats=[[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    world.object_views = {"bowl2": _fake_object_view((-1.0, -0.5, 0.5))}
    world.arms = _FakeArmsForArmSide(
        left_succeeded=True, right_succeeded=False
    )

    assert world._select_arm_side("bowl2") == "left"


def test_select_arm_side_bowl2_case_selects_left():
    # GATE T3 (plans/LOOP_PROMPT_VM_A_REV9.md): "the bowl2 case selects
    # left." Real recorded shape, plans/VM_B_LOG.md 2026-08-02: RIGHT arm
    # 0/1600 + 0/1200 + 0/1200 real IK-ok ticks on this exact ER-derived
    # pose; LEFT 191/191 + 1200/1200 + 1200/1200, object_follows_ee: true.
    # Geometry alone (the old metric) would have picked RIGHT here --
    # object placed deliberately closer to right_base, same as the
    # distance-only test above -- so this only passes because feasibility
    # now overrides distance, not by accident of setup.
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.robot = _fake_robot(
        body_names=["root", "left_base", "right_base"],
        positions=[[0.0, 0.0, 0.0], [-1.0, 0.05, 0.5], [-1.0, -0.05, 0.5]],
        quats=[[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    world.object_views = {"bowl2": _fake_object_view((-1.0, -0.5, 0.5))}
    world.arms = _FakeArmsForArmSide(
        left_succeeded=True, right_succeeded=False
    )

    assert world._select_arm_side("bowl2") == "left"


def test_select_arm_side_falls_back_to_distance_when_feasibility_ties():
    # Both sides feasible (or both infeasible) is not a decision the
    # feasibility query can make -- must fall back to the old
    # distance-based tie-break rather than picking arbitrarily.
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.robot = _fake_robot(
        body_names=["root", "left_base", "right_base"],
        positions=[[0.0, 0.0, 0.0], [-1.0, 0.05, 0.5], [-1.0, -0.05, 0.5]],
        quats=[[1.0, 0.0, 0.0, 0.0]] * 3,
    )
    # Object closer to left_base, as in the geometric-preference test.
    world.object_views = {"bowl2": _fake_object_view((-1.0, 0.5, 0.5))}
    world.arms = _FakeArmsForArmSide(left_succeeded=True, right_succeeded=True)

    assert world._select_arm_side("bowl2") == "left"


def test_carry_object_to_ignores_nearer_side_selection_when_flag_is_off():
    # "trivially revertible": select_nearer_arm_side defaults False, so
    # carry_object_to() must resolve to plain "right" without ever calling
    # _select_arm_side -- confirmed by making that method raise if invoked.
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    assert world.select_nearer_arm_side is False

    captured = {}

    def _fake_push(side, object_name, x, y, z, **p):
        captured["side"] = side
        return {"scored": False}

    world._push_object_to = _fake_push

    def _boom(_object_name):
        raise AssertionError("_select_arm_side must not run when flag is off")

    world._select_arm_side = _boom

    world.carry_object_to("bowl2", 0.0, 0.0, method="base_carry")
    assert captured["side"] == "right"


def test_carry_object_to_uses_nearer_side_selection_when_flag_is_on():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(
        simulation_app=None, select_nearer_arm_side=True
    )

    captured = {}

    def _fake_push(side, object_name, x, y, z, **p):
        captured["side"] = side
        return {"scored": False}

    world._push_object_to = _fake_push
    world._select_arm_side = lambda object_name: "left"

    world.carry_object_to("bowl2", 0.0, 0.0, method="base_carry")
    assert captured["side"] == "left"


def test_carry_object_to_explicit_side_overrides_selection():
    # An explicit side= kwarg must win even with the flag on -- selection
    # is a default, not a forced override of a caller's own choice.
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(
        simulation_app=None, select_nearer_arm_side=True
    )

    captured = {}

    def _fake_push(side, object_name, x, y, z, **p):
        captured["side"] = side
        return {"scored": False}

    world._push_object_to = _fake_push

    def _boom(_object_name):
        raise AssertionError("explicit side= must skip selection entirely")

    world._select_arm_side = _boom

    world.carry_object_to("bowl2", 0.0, 0.0, method="base_carry", side="right")
    assert captured["side"] == "right"


ALL = [v for k, v in dict(globals()).items() if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for fn in ALL:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL) - failures}/{len(ALL)} passed")
    raise SystemExit(1 if failures else 0)


def _push_object_to_harness(arm_base_norm):
    """Drives the REAL production _push_object_to() up to the new Q2
    pre-flight reach-limit gate, faking only the seams push_approach's
    call site does not itself exercise (navigate_to/_rotate_to/_stance_for
    -- the base repositioning that happens before the arm ever reaches for
    the object -- and arms.reach itself, so this test can tell whether it
    was called at all rather than needing a real IK solve). `arm_base_norm`
    controls what world._arm_base_relative() reports for approach_target,
    the exact seam SYNC 22/23's fix gates on.
    """
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world.robot = None
    world.record_video = False
    world._active_object = None
    world.object_views = {"cup": _fake_object_view((1.0, 0.0, 0.75))}
    world._m = {
        "ramp_arm_pose": lambda robot, pose, **kw: None,
        "TRANSIT_ARM_POSE": {},
        "_quaternion_from_rpy": lambda r, p, y: (1.0, 0.0, 0.0, 0.0),
        # `_push_object_to` reaches its stance via
        # `navigate_to_avoiding_island`, which looks this up in `_m`. The
        # real function is pure and CPU-safe (no Isaac import), so wire the
        # real one rather than a stub -- the routing geometry is exactly
        # what the island-crossing bug lived in, and a stub would hide it.
        "route_avoiding_island": route_avoiding_island,
    }
    world.navigate_to = lambda *a, **kw: {"terminal_error_m": 0.0}
    world._rotate_to = lambda yaw: None
    world._stance_for = (
        lambda object_xy, approach, contact_z=None, stance_radius_m=None, **kw: (
            (0.0, 0.0),
            0.0,
        )
    )
    world.arms = _FakeArmsForGraspFrame((1.0, 0.0, 0.75))
    world.arms.reach_calls = 0

    def _fake_reach(*a, **kw):
        # Only the FIRST reach() call (push_approach's own) matters to this
        # test -- raise past it so a within-limit run exits cleanly via the
        # method's own existing ValueError handling instead of needing to
        # also stub the manual gentle-ramp tick loop further down (a
        # separate code path this fix does not touch).
        world.arms.reach_calls += 1
        if world.arms.reach_calls == 1:
            return True
        raise ValueError("test stub: stop after push_approach's reach()")

    world.arms.reach = _fake_reach
    world.arms.release = lambda *a, **kw: None
    world._arm_base_relative = lambda side, target: (
        [0.0, 0.0, 0.0],
        arm_base_norm,
    )

    result = world._push_object_to("right", "cup", 1.5, 0.0, 0.75)
    return world, result


def test_push_approach_bails_before_reach_when_target_exceeds_reach_limit():
    # Q2 (SYNC 22/23, task3-submission @ 659d792): real GPU evidence traced
    # cup's counter-edge knockoff to a push_approach target 0.037m past the
    # measured ~0.855m FR3 reach ceiling -- the arm strained at/beyond its
    # kinematic limit, directly above the object, for the full 8s reach()
    # budget before the object left the counter. The fix gates on the same
    # ceiling BEFORE calling reach() at all. Norm here (0.9) mirrors the
    # real traced value (0.8921).
    world, result = _push_object_to_harness(arm_base_norm=0.9)
    assert world.arms.reach_calls == 0, (
        "reach() must not be called once the pre-flight check already "
        f"knows the target is past the reach ceiling: {result}"
    )
    assert result == {
        "scored": False,
        "reason": "push_approach_beyond_reach_limit",
    }, result


def test_push_approach_proceeds_normally_when_target_within_reach_limit():
    # Same harness, a target comfortably inside the reach ceiling -- the
    # gate must not fire and the real reach() call must still happen so
    # in-range pushes are unaffected by this fix.
    world, result = _push_object_to_harness(arm_base_norm=0.5)
    assert world.arms.reach_calls >= 1, (
        f"an in-range target must still reach reach(): {result}"
    )


def test_push_approach_and_standoff_pass_zero_success_bail_ticks():
    # T3 (plans/LOOP_PROMPT_VM_A_REV5.md): T2 traced a live episode where
    # push_approach's reach() call -- unlike push_contact's, which already
    # passes zero_success_bail_ticks (GATE B1, handoff sec 78/80) -- ran
    # its full budget after its IK started diverging and swept the arm
    # through the object, flinging it off the table. This guard's real
    # effect (does 150 consecutive IK failures actually bail early) can
    # only be observed on GPU; this test is the CPU-checkable half -- that
    # the kwarg reaches arms.reach() at all for BOTH push_approach and
    # push_standoff, which would fail loudly (AttributeError: unexpected
    # keyword) if either call site regressed.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world.robot = None
    world.record_video = False
    world._active_object = None
    world.object_views = {"cup": _fake_object_view((1.0, 0.0, 0.75))}
    world._m = {
        "ramp_arm_pose": lambda robot, pose, **kw: None,
        "TRANSIT_ARM_POSE": {},
        "_quaternion_from_rpy": lambda r, p, y: (1.0, 0.0, 0.0, 0.0),
        # `_push_object_to` reaches its stance via
        # `navigate_to_avoiding_island`, which looks this up in `_m`. The
        # real function is pure and CPU-safe (no Isaac import), so wire the
        # real one rather than a stub -- the routing geometry is exactly
        # what the island-crossing bug lived in, and a stub would hide it.
        "route_avoiding_island": route_avoiding_island,
    }
    world.navigate_to = lambda *a, **kw: {"terminal_error_m": 0.0}
    world._rotate_to = lambda yaw: None
    world._stance_for = (
        lambda object_xy, approach, contact_z=None, stance_radius_m=None, **kw: (
            (0.0, 0.0),
            0.0,
        )
    )
    world.arms = _FakeArmsForGraspFrame((1.0, 0.0, 0.75))
    world.arms.release = lambda *a, **kw: None
    world._arm_base_relative = lambda side, target: ([0.0, 0.0, 0.0], 0.5)

    captured_kwargs = []

    def _fake_reach(*a, **kw):
        captured_kwargs.append(kw)
        if len(captured_kwargs) >= 2:  # stop after push_standoff's call
            raise ValueError("test stub: stop after push_standoff's reach()")
        return True

    world.arms.reach = _fake_reach

    world._push_object_to("right", "cup", 1.5, 0.0, 0.75)

    assert len(captured_kwargs) == 2, captured_kwargs
    push_approach_kwargs, push_standoff_kwargs = captured_kwargs
    assert push_approach_kwargs.get("zero_success_bail_ticks") == 150, (
        push_approach_kwargs
    )
    assert push_standoff_kwargs.get("zero_success_bail_ticks") == 150, (
        push_standoff_kwargs
    )


def test_stance_validated_against_bare_contact_height_not_approach_height():
    # T5a (plans/LOOP_PROMPT_VM_A_REV5.md, ADDED 2026-08-04): validating
    # the stance against `contact_z + approach_clearance` (the real,
    # higher point push_approach reaches for) instead of the bare contact
    # height was tried and GPU-verified REFUTED (2026-08-04 ~12:00 UTC,
    # plans/SYNC.md) -- navigate_to failed to arrive at the chosen stance
    # in every attempt, a regression against T4's confirmed-good baseline.
    # Reverted. This test now guards against silently re-introducing that
    # regression: _stance_for's contact_z argument must stay the bare
    # object height, not the higher approach height.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world.robot = None
    world.record_video = False
    world._active_object = None
    object_z = 0.75
    world.object_views = {"cup": _fake_object_view((1.0, 0.0, object_z))}
    world._m = {
        "ramp_arm_pose": lambda robot, pose, **kw: None,
        "TRANSIT_ARM_POSE": {},
        "_quaternion_from_rpy": lambda r, p, y: (1.0, 0.0, 0.0, 0.0),
        # `_push_object_to` reaches its stance via
        # `navigate_to_avoiding_island`, which looks this up in `_m`. The
        # real function is pure and CPU-safe (no Isaac import), so wire the
        # real one rather than a stub -- the routing geometry is exactly
        # what the island-crossing bug lived in, and a stub would hide it.
        "route_avoiding_island": route_avoiding_island,
    }
    world.navigate_to = lambda *a, **kw: {"terminal_error_m": 0.0}
    world._rotate_to = lambda yaw: None
    captured_stance_for_kwargs = {}

    def _fake_stance_for(
        object_xy,
        approach,
        contact_z=None,
        stance_radius_m=None,
        stance_max_radius_m=None,
    ):
        captured_stance_for_kwargs["contact_z"] = contact_z
        captured_stance_for_kwargs["stance_radius_m"] = stance_radius_m
        captured_stance_for_kwargs["stance_max_radius_m"] = (
            stance_max_radius_m
        )
        return (0.0, 0.0), 0.0

    world._stance_for = _fake_stance_for
    world.arms = _FakeArmsForGraspFrame((1.0, 0.0, object_z))
    world.arms.release = lambda *a, **kw: None

    def _fake_reach(*a, **kw):
        raise ValueError("test stub: stop right after the stance call")

    world.arms.reach = _fake_reach

    world._push_object_to("right", "cup", 1.5, 0.0, object_z)

    assert math.isclose(
        captured_stance_for_kwargs["contact_z"],
        object_z,
        abs_tol=1e-9,
    ), captured_stance_for_kwargs
    # R7 T2 (plans/SYNC.md 2026-08-04 ~19:48 UTC): push must request the
    # sweep-derived closer stance radius by default, not the unaffected
    # grasp-calibrated STANCE_REACH_RADIUS_M.
    assert math.isclose(
        captured_stance_for_kwargs["stance_radius_m"],
        world_isaac_mod.PUSH_STANCE_RADIUS_M,
        abs_tol=1e-9,
    ), captured_stance_for_kwargs


def test_push_stance_navigate_budget_s_is_overridable_and_defaults_unchanged():
    # Real, well-evidenced next lever (plans/SYNC.md 2026-08-04 ~14:45 UTC
    # correction): navigate_to has never arrived at a curobo_stance_for
    # candidate in any recorded run since T4 -- T4's fix picks candidates
    # ~2-3m away and the stance-approach navigate_to call's budget_s was
    # hardcoded at 25.0s. Made it a p.get()-overridable kwarg
    # (push_stance_navigate_budget_s) so it can be A/B'd without a code
    # change. This asserts: (a) the default stays 25.0 (no behavior
    # change for existing callers), (b) an override actually reaches
    # navigate_to.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    def _make_world(object_z=0.75):
        world = world_isaac_mod.IsaacWorld(simulation_app=None)
        world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
        world.adapter = types.SimpleNamespace(
            pose=lambda: Pose2D(0.0, 0.0, 0.0)
        )
        world.robot = None
        world.record_video = False
        world._active_object = None
        world.object_views = {"cup": _fake_object_view((1.0, 0.0, object_z))}
        world._m = {
            "ramp_arm_pose": lambda robot, pose, **kw: None,
            "TRANSIT_ARM_POSE": {},
            "_quaternion_from_rpy": lambda r, p, y: (1.0, 0.0, 0.0, 0.0),
            # See the sibling harness above: navigate_to_avoiding_island
            # looks this up in `_m`.
            "route_avoiding_island": route_avoiding_island,
        }
        captured_navigate_kwargs = {}

        def _fake_navigate_to(*a, **kw):
            captured_navigate_kwargs.update(kw)
            return {"terminal_error_m": 0.0}

        world.navigate_to = _fake_navigate_to
        world._rotate_to = lambda yaw: None
        world._stance_for = (
            lambda object_xy, approach, contact_z=None, stance_radius_m=None, **kw: (
                (0.0, 0.0),
                0.0,
            )
        )
        world.arms = _FakeArmsForGraspFrame((1.0, 0.0, object_z))
        world.arms.release = lambda *a, **kw: None

        def _fake_reach(*a, **kw):
            raise ValueError("test stub: stop right after navigate_to")

        world.arms.reach = _fake_reach
        return world, captured_navigate_kwargs, object_z

    world, captured, object_z = _make_world()
    world._push_object_to("right", "cup", 1.5, 0.0, object_z)
    assert captured.get("budget_s") == 25.0, captured

    world2, captured2, object_z2 = _make_world()
    world2._push_object_to(
        "right", "cup", 1.5, 0.0, object_z2, push_stance_navigate_budget_s=45.0
    )
    assert captured2.get("budget_s") == 45.0, captured2


def _push_object_to_perception_harness(
    object_name, push_perception_targets, cache
):
    """Same production _push_object_to() harness as Q2's, extended to
    exercise Q3's perception-cache override. Pre-seeds
    world._perception_push_target_cache and marks
    _perception_push_attempted so the (GPU/API-dependent) ER-calling
    machinery in _ensure_perception_push_targets() never runs -- this
    tests the WIRING (does the cache actually change contact_z, and only
    for PUSH_PERCEPTION_OBJECTS with the flag on), not the ER call itself
    (already covered by task3_autonomy/tests/test_perception_targets.py
    and the live GATE N2/N4 probes).
    """
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=0.005))
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world.robot = None
    world.record_video = False
    world._active_object = None
    world.object_views = {
        object_name: _fake_object_view((1.0, 0.0, 0.75)),
    }
    world._m = {
        "ramp_arm_pose": lambda robot, pose, **kw: None,
        "TRANSIT_ARM_POSE": {},
        "_quaternion_from_rpy": lambda r, p, y: (1.0, 0.0, 0.0, 0.0),
        # `_push_object_to` reaches its stance via
        # `navigate_to_avoiding_island`, which looks this up in `_m`. The
        # real function is pure and CPU-safe (no Isaac import), so wire the
        # real one rather than a stub -- the routing geometry is exactly
        # what the island-crossing bug lived in, and a stub would hide it.
        "route_avoiding_island": route_avoiding_island,
    }
    world.navigate_to = lambda *a, **kw: {"terminal_error_m": 0.0}
    world._rotate_to = lambda yaw: None
    world._stance_for = (
        lambda object_xy, approach, contact_z=None, stance_radius_m=None, **kw: (
            (0.0, 0.0),
            0.0,
        )
    )
    world.arms = _FakeArmsForGraspFrame((1.0, 0.0, 0.75))
    world.arms.release = lambda *a, **kw: None
    world._arm_base_relative = lambda side, target: ([0.0, 0.0, 0.0], 0.5)

    captured = {"calls": 0}

    def _fake_reach(side, position, *a, **kw):
        captured["calls"] += 1
        if captured["calls"] == 1:
            captured["position"] = position
            return True
        # Only push_approach's own reach() call (the first) matters here --
        # stop right after it, same trick Q2's harness uses, instead of
        # also stubbing the manual gentle-ramp tick loop further down (an
        # unrelated code path this test does not touch).
        raise ValueError("test stub: stop after push_approach's reach()")

    world.arms.reach = _fake_reach

    world.push_perception_targets = push_perception_targets
    world._perception_push_attempted = True  # skip the real ER call path
    world._perception_push_target_cache = dict(cache)

    world._push_object_to("right", object_name, 1.5, 0.0, 0.75)
    return captured["position"]


def test_push_contact_z_uses_perception_cache_when_eligible_and_flag_on():
    # Q3 (SYNC 22-24): bowl2 is in PUSH_PERCEPTION_OBJECTS. A cached
    # perception candidate (z=0.9, well above the object's live 0.75) must
    # override contact_z when the flag is on.
    position = _push_object_to_perception_harness(
        "bowl2",
        push_perception_targets=True,
        cache={"bowl2": (1.0, 0.0, 0.9)},
    )
    # contact_z = perception z (0.9) + approach_clearance default (0.15).
    assert position[2] == 1.05, position


def test_push_contact_z_ignores_perception_cache_when_flag_off():
    # Same cache present, but the flag is off -- must fall back to the
    # existing live-geometry contact_z untouched.
    position = _push_object_to_perception_harness(
        "bowl2",
        push_perception_targets=False,
        cache={"bowl2": (1.0, 0.0, 0.9)},
    )
    # contact_z = live obj z (0.75) + approach_clearance default (0.15).
    assert position[2] == 0.9, position


def test_push_contact_z_ignores_perception_cache_for_object_outside_scope():
    # cup is explicitly OUT of PUSH_PERCEPTION_OBJECTS (N3 refuted
    # perception for cup) -- a cache entry for it must never apply even
    # with the flag on.
    position = _push_object_to_perception_harness(
        "cup",
        push_perception_targets=True,
        cache={"cup": (1.0, 0.0, 0.9)},
    )
    assert position[2] == 0.9, position


def test_push_contact_bails_before_reach_when_target_exceeds_reach_limit():
    # Q4 (SYNC 22-25): real GPU evidence (task3-submission, Q3 verification
    # run) traced push_contact's own near-zero success rate to the SAME
    # mechanism Q2 fixed for push_approach -- 3 of 4 real push_contact
    # attempts had target_norm_from_arm_base_m past the 0.855m ceiling
    # (1.0982, 1.2296, 0.897) and NONE of the 4 converged, meaning
    # push_drive (the phase that actually drives the base and displaces
    # the object) never ran in any observed episode. This drives the real
    # production _push_object_to() through push_approach and the gentle
    # ramp (both must succeed, matching a realistic in-range approach)
    # and only fails the LATER push_contact gate -- proving the new check
    # is scoped to push_contact specifically, not a blanket block.
    import types

    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.navigation import Pose2D

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world.sim = types.SimpleNamespace(
        cfg=types.SimpleNamespace(dt=0.005), step=lambda render=None: None
    )
    world.adapter = types.SimpleNamespace(pose=lambda: Pose2D(0.0, 0.0, 0.0))
    world.scene = types.SimpleNamespace(
        write_data_to_sim=lambda: None, update=lambda dt: None
    )
    world.robot = None
    world.record_video = False
    world._active_object = None
    world.object_views = {"bowl2": _fake_object_view((1.0, 0.0, 0.75))}
    world._m = {
        "ramp_arm_pose": lambda robot, pose, **kw: None,
        "TRANSIT_ARM_POSE": {},
        "_quaternion_from_rpy": lambda r, p, y: (1.0, 0.0, 0.0, 0.0),
        "disable_robot_external_wrenches": lambda robot: None,
        # See the sibling harnesses above: navigate_to_avoiding_island
        # looks this up in `_m`.
        "route_avoiding_island": route_avoiding_island,
    }
    world.navigate_to = lambda *a, **kw: {"terminal_error_m": 0.0}
    world._rotate_to = lambda yaw: None
    world._stance_for = (
        lambda object_xy, approach, contact_z=None, stance_radius_m=None, **kw: (
            (0.0, 0.0),
            0.0,
        )
    )
    world.record_video = False
    world.arms = _FakeArmsForGraspFrame((1.0, 0.0, 0.75))
    world.arms.release = lambda *a, **kw: None
    world.arms.reach_calls = 0

    def _fake_reach(*a, **kw):
        world.arms.reach_calls += 1
        return True

    world.arms.reach = _fake_reach

    _command_result = types.SimpleNamespace(
        left_succeeded=True, right_succeeded=True
    )
    world.arms.set_arm_target = lambda *a, **kw: None
    world.arms.command = lambda: _command_result

    # _reach_limit_exceeded is called exactly twice per attempt in the
    # code (push_approach's own target, then push_contact's -- the
    # ramp_start reach() in between is NOT gated, matching production):
    # False the first time (approach is in range), True the second
    # (isolating this test to the push_contact call site specifically).
    calls = {"n": 0}

    def _fake_reach_limit_exceeded(side, target):
        calls["n"] += 1
        return calls["n"] >= 2

    world._reach_limit_exceeded = _fake_reach_limit_exceeded

    result = world._push_object_to("right", "bowl2", 1.5, 0.0, 0.75)

    assert result == {
        "scored": False,
        "reason": "push_contact_beyond_reach_limit",
    }, result
    # approach + ramp_start both succeeded (2 real reach() calls) before
    # the gate correctly stopped it from ever attempting push_contact's
    # own reach() (which would have been the 3rd call).
    assert world.arms.reach_calls == 2, world.arms.reach_calls


# ---------------------------------------------------------------------- #
# R9 T4: reach_and_grasp_ranked -- iterate ranked candidates, fall through
# on failure, hardcoded pose as last resort.
# ---------------------------------------------------------------------- #


def _ranked_pair(candidate_id, side, rank, feasible=True):
    from task3_autonomy.grasp_contract import GraspCandidate, RankedGrasp

    candidate = GraspCandidate(
        id=candidate_id,
        position=(-4.0 - candidate_id * 0.01, -1.5, 0.83),
        yaw_rad=0.7854,
        tilt_rad=0.0,
        source="er",
        label="rim",
        confidence=0.8,
    )
    ranked = RankedGrasp(
        candidate_id=candidate_id,
        side=side,
        stance_xy=(-3.77, -0.82),
        stance_yaw=3.142,
        feasible=feasible,
        ik_margin=0.1,
        rank=rank,
    )
    return ranked, candidate


_UNMOVED_OBJECT_POSE = (-4.0, -1.5, 0.83)


def test_reach_and_grasp_ranked_tries_candidate_2_when_candidate_1_fails():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._load_ranked_grasp_plan = lambda object_name, side: (
        [_ranked_pair(0, "left", rank=0), _ranked_pair(1, "left", rank=1)],
        _UNMOVED_OBJECT_POSE,
    )
    world.object_position = lambda object_name: _UNMOVED_OBJECT_POSE
    memory_calls = []
    world._append_grasp_attempt_memory = lambda *a, **kw: memory_calls.append(
        kw
    )

    reach_calls = []

    def _fake_reach(side, object_name, **p):
        reach_calls.append(p.get("grasp_xyz_override"))
        return {"position_error_m": 0.05}

    grasp_outcomes = iter(
        [False, True]
    )  # candidate 1 fails, candidate 2 holds

    def _fake_grasp(side, object_name, **p):
        return {"object_follows_ee": next(grasp_outcomes)}

    world.reach = _fake_reach
    world.grasp = _fake_grasp
    # 2026-08-19 (owner directive): `held` is telemetry AND an independent
    # wrist-camera confirmation, not telemetry alone. This test is about
    # candidate RETRY ORDERING, so pin the camera gate open and let the
    # grasp outcomes drive the result; the gate itself is pinned by
    # test_ranked_grasp_requires_camera_confirmation_not_telemetry_alone.
    world.verify_grasp_by_wrist_camera = lambda side, object_name: {
        "verified": True,
        "pixel_frac": 1.0,
    }

    result = world.reach_and_grasp_ranked("left", "bowl2")

    assert len(reach_calls) == 2, (
        "candidate 2 must be tried after candidate 1 fails"
    )
    assert result["used_ranked_candidate_id"] == 1
    assert result["fell_back_to_hardcoded"] is False
    assert len(result["attempts"]) == 2
    assert [c["candidate_id"] for c in memory_calls] == [0, 1]
    assert memory_calls[0]["object_follows_ee"] is False
    assert memory_calls[1]["object_follows_ee"] is True


def test_reach_and_grasp_ranked_falls_back_to_hardcoded_when_no_ranked_plan():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._load_ranked_grasp_plan = lambda object_name, side: ([], None)
    memory_calls = []
    world._append_grasp_attempt_memory = lambda *a, **kw: memory_calls.append(
        kw
    )

    reach_calls = []

    def _fake_reach(side, object_name, **p):
        reach_calls.append(p)
        return {"position_error_m": 0.02}

    def _fake_grasp(side, object_name, **p):
        return {"object_follows_ee": True}

    world.reach = _fake_reach
    world.grasp = _fake_grasp

    result = world.reach_and_grasp_ranked("right", "cup")

    assert len(reach_calls) == 1
    # No overrides -- byte-identical to calling reach()/grasp() directly.
    assert "grasp_xyz_override" not in reach_calls[0]
    assert "stance_xy_override" not in reach_calls[0]
    assert result["fell_back_to_hardcoded"] is True
    assert result["used_ranked_candidate_id"] is None
    # Pure no-ranked-file case (cup): nothing to attribute a memory entry
    # to, so no memory write -- avoids noise, per the method's own
    # docstring.
    assert memory_calls == []


def test_reach_and_grasp_ranked_logs_fallback_attempt_when_plan_exhausted():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._load_ranked_grasp_plan = lambda object_name, side: (
        [_ranked_pair(0, "left", rank=0)],
        _UNMOVED_OBJECT_POSE,
    )
    world.object_position = lambda object_name: _UNMOVED_OBJECT_POSE
    memory_calls = []
    world._append_grasp_attempt_memory = lambda *a, **kw: memory_calls.append(
        kw
    )
    world.reach = lambda side, object_name, **p: {"position_error_m": 0.5}
    world.grasp = lambda side, object_name, **p: {"object_follows_ee": False}

    result = world.reach_and_grasp_ranked("left", "bowl2")

    assert result["fell_back_to_hardcoded"] is True
    # One memory entry for the failed ranked candidate, one for the
    # hardcoded-pose fallback that was actually tried.
    assert [c["candidate_id"] for c in memory_calls] == [0, -1]


def test_reach_and_grasp_ranked_respects_max_attempts_cap():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._load_ranked_grasp_plan = lambda object_name, side: (
        [_ranked_pair(i, "left", rank=i) for i in range(6)],
        _UNMOVED_OBJECT_POSE,
    )
    world.object_position = lambda object_name: _UNMOVED_OBJECT_POSE
    world._append_grasp_attempt_memory = lambda *a, **kw: None

    reach_calls = []
    world.reach = lambda side, object_name, **p: (
        reach_calls.append(1),
        {"position_error_m": 0.5},
    )[1]
    world.grasp = lambda side, object_name, **p: {"object_follows_ee": False}

    world.reach_and_grasp_ranked("left", "bowl2", max_attempts=3)

    # 3 capped ranked attempts + 1 final hardcoded-pose fallback call.
    assert len(reach_calls) == 4


# ---------------------------------------------------------------------- #
# REV12 T5: reach_and_grasp_ranked re-anchors to the LIVE object pose
# before each attempt, and abandons (without calling reach()/grasp()) a
# candidate whose object has moved too far or dropped to the floor.
# ---------------------------------------------------------------------- #


def test_reach_and_grasp_ranked_abandons_on_floor_drop_without_calling_reach():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._load_ranked_grasp_plan = lambda object_name, side: (
        [_ranked_pair(0, "left", rank=0)],
        _UNMOVED_OBJECT_POSE,
    )
    # Object dropped 0.5m in z since the candidate was generated -- on
    # the floor.
    dropped_pose = (
        _UNMOVED_OBJECT_POSE[0],
        _UNMOVED_OBJECT_POSE[1],
        _UNMOVED_OBJECT_POSE[2] - 0.5,
    )
    world.object_position = lambda object_name: dropped_pose
    memory_calls = []
    world._append_grasp_attempt_memory = lambda *a, **kw: memory_calls.append(
        kw
    )

    reach_calls = []
    world.reach = lambda side, object_name, **p: (
        reach_calls.append(1),
        {"position_error_m": 0.02},
    )[1]
    world.grasp = lambda side, object_name, **p: {"object_follows_ee": True}

    result = world.reach_and_grasp_ranked("left", "bowl2")

    # The ranked candidate must be abandoned (never reached for) -- only
    # the final hardcoded-pose fallback calls reach()/grasp().
    assert len(reach_calls) == 1
    assert result["attempts"][0]["abandoned"] is True
    assert result["attempts"][0]["reanchor_action"] == "abandon_floor"
    assert memory_calls[0]["source"] == "reach_and_grasp_ranked_abandon_floor"
    assert memory_calls[0]["object_follows_ee"] is False


def test_reach_and_grasp_ranked_abandons_on_absurd_jump():
    """A jump larger than the sanity ceiling but still inside the coarse
    scene bounds -- the actual 223m spoon2 fling lands outside scene
    bounds entirely (covered by grasp_reanchor's own out-of-bounds
    tests); this proves the jump-specific check independently."""
    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.grasp_reanchor import MAX_SANE_DELTA_M

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._load_ranked_grasp_plan = lambda object_name, side: (
        [_ranked_pair(0, "left", rank=0)],
        _UNMOVED_OBJECT_POSE,
    )
    flung_pose = (
        _UNMOVED_OBJECT_POSE[0] + 1.0 + MAX_SANE_DELTA_M,
        _UNMOVED_OBJECT_POSE[1],
        _UNMOVED_OBJECT_POSE[2],
    )
    world.object_position = lambda object_name: flung_pose
    world._append_grasp_attempt_memory = lambda *a, **kw: None

    reach_calls = []
    world.reach = lambda side, object_name, **p: (
        reach_calls.append(1),
        {"position_error_m": 0.02},
    )[1]
    world.grasp = lambda side, object_name, **p: {"object_follows_ee": True}

    result = world.reach_and_grasp_ranked("left", "bowl2")

    assert len(reach_calls) == 1  # only the hardcoded-pose fallback
    assert result["attempts"][0]["reanchor_action"] == "abandon_jump"


def test_reach_and_grasp_ranked_translates_candidate_on_routine_drift():
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    ranked_entry, candidate = _ranked_pair(0, "left", rank=0)
    world._load_ranked_grasp_plan = lambda object_name, side: (
        [(ranked_entry, candidate)],
        _UNMOVED_OBJECT_POSE,
    )
    drifted_pose = (
        _UNMOVED_OBJECT_POSE[0] + 0.02,
        _UNMOVED_OBJECT_POSE[1] - 0.01,
        _UNMOVED_OBJECT_POSE[2],
    )
    world.object_position = lambda object_name: drifted_pose
    world._append_grasp_attempt_memory = lambda *a, **kw: None

    reach_targets = []

    def _fake_reach(side, object_name, **p):
        reach_targets.append(p.get("grasp_xyz_override"))
        return {"position_error_m": 0.02}

    world.reach = _fake_reach
    world.grasp = lambda side, object_name, **p: {"object_follows_ee": True}

    world.reach_and_grasp_ranked("left", "bowl2")

    expected = (
        candidate.position[0] + 0.02,
        candidate.position[1] - 0.01,
        candidate.position[2],
    )
    for got, want in zip(reach_targets[0], expected):
        assert abs(got - want) < 1e-9


def test_load_ranked_grasp_plan_filters_by_side_sorts_by_rank(monkeypatch):
    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.grasp_contract import CandidateFile, RankedFile

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    left0, cand0 = _ranked_pair(0, "left", rank=1)
    left1, cand1 = _ranked_pair(1, "left", rank=0)
    right0, _cand_right = _ranked_pair(0, "right", rank=2, feasible=False)

    candidate_file = CandidateFile(
        object="bowl2",
        object_pose=(-4.0, -1.5, 0.74),
        generated_utc="2026-08-05T00:00:00Z",
        candidates=(cand0, cand1),
    )
    ranked_file = RankedFile(object="bowl2", ranked=(left0, left1, right0))

    monkeypatch.setattr(
        world_isaac_mod, "load_candidates", lambda object_name: candidate_file
    )
    monkeypatch.setattr(
        world_isaac_mod, "load_ranked", lambda object_name: ranked_file
    )

    pairs, object_pose = world._load_ranked_grasp_plan("bowl2", "left")

    assert [c.id for _r, c in pairs] == [1, 0]  # sorted by rank: 0 before 1
    assert object_pose == (-4.0, -1.5, 0.74)


def test_load_ranked_grasp_plan_returns_empty_on_missing_files(monkeypatch):
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)

    def _raise(object_name):
        raise FileNotFoundError(object_name)

    monkeypatch.setattr(world_isaac_mod, "load_candidates", _raise)

    assert world._load_ranked_grasp_plan("bowl2", "left") == ([], None)


def test_load_ranked_grasp_plan_returns_empty_on_contract_error(monkeypatch):
    import task3_pipeline.world_isaac as world_isaac_mod
    from task3_autonomy.grasp_contract import GraspContractError

    world = world_isaac_mod.IsaacWorld(simulation_app=None)

    def _raise(object_name):
        raise GraspContractError("malformed")

    monkeypatch.setattr(world_isaac_mod, "load_candidates", _raise)

    assert world._load_ranked_grasp_plan("bowl2", "left") == ([], None)


def test_carry_object_to_commands_the_arm_every_tick():
    """The carry loop must WRITE its targets, not just set them.

    `set_arm_target_relative` and `set_gripper` mutate the tracker only;
    `arms.command()` is what solves IK and calls
    `set_joint_position_target`. Without it the arm stays frozen in joint
    space for the whole drive while the base moves out from under it.

    Measured, allobj_3: a genuinely held spoon2 (latched at 0.4598 rad,
    `object_follows_ee True`) read 1.0028 rad -- the gripper's own max-open
    stop -- one sample into the carry, with the object left on the counter.
    """
    import types

    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)

    calls: list[str] = []
    ticks = {"n": 0}

    class _Arms:
        def arm_pose_relative(self, side):
            return types.SimpleNamespace(
                position=(0.4, 0.1, 0.3), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)
            )

        def set_arm_target_relative(self, side, position, quat_wxyz):
            calls.append("target")

        def set_gripper(self, side, position_rad):
            calls.append(f"gripper:{position_rad}")

        def command(self):
            calls.append("command")

        def gripper_position(self, side):
            return 0.4598

        def ee_world_poses(self):
            return (((0.0, 0.0, 0.8), (1.0, 0.0, 0.0, 0.0)),) * 2

        def release(self, side, *, step, dt, timeout_s):
            return True

    class _Adapter:
        def pose(self):
            # Must actually move, or ProgressWatchdog reports a stall and
            # the loop breaks before proving anything.
            return types.SimpleNamespace(x=0.1 * ticks["n"], y=0.0, yaw=0.0)

        def apply_twist(self, vx, vy, **kw):
            return None

    class _Skill:
        def __init__(self, *a, **kw):
            self._n = 0

        def compute(self, pose):
            self._n += 1
            return 0.1, 0.0, self._n >= 5

    world.arms = _Arms()
    world.adapter = _Adapter()
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=1.0 / 60.0))
    world._m = dict(world._m or {})
    world._m["NavigateTo"] = _Skill
    world._held = "spoon2"
    world._held_side = "left"
    world._held_gripper_rad = 0.4598
    world.object_position = lambda name: (0.0, 0.0, 0.8)

    def _tick():
        ticks["n"] += 1
        world._tick_count = ticks["n"]

    world._tick = _tick
    world._log_phase = lambda *a, **kw: None

    world.carry_object_to("spoon2", -1.6, 1.0)

    assert "command" in calls, (
        "carry_object_to never called arms.command(), so nothing it set "
        "each tick ever reached the robot"
    )
    # One command() per commanded target, or the arm lags the base.
    assert calls.count("command") == calls.count("target")
    # And the latched hold must be re-sent alongside it.
    assert any(c.startswith("gripper:") for c in calls)


def test_stage_worker_prints_a_traceback_for_base_exceptions():
    """A BaseException in a stage plan must be named, not silent.

    Measured, allobj_4: the worker died at the head-camera capture and the
    episode reported only "WorkerDiedError: ... died before put(), e.g. via
    an uncaught BaseException" -- no traceback, no diagnosis, four minutes
    of GPU spent. `except Exception` cannot see that class of failure.

    The WorkerDiedError outcome itself is a separate, tested contract and
    is deliberately left alone here; this pins only the diagnosis.
    """
    import io
    from contextlib import redirect_stdout

    import task3_pipeline.orchestrator as orch_mod

    def dying_plan(runner, world):
        raise SystemExit("camera capture died")

    original_plans = dict(orch_mod.STAGE_PLANS)
    orch_mod.STAGE_PLANS[1] = dying_plan
    try:
        world = MockWorld(seed=1, head_placement="a")
        pipe = Task3Pipeline(
            world, memory_path=None, stage_wallclock_budget_s=5.0
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = pipe.run_episode(seed=1, head_placement="a", order=(1,))
        out = buf.getvalue()
    finally:
        orch_mod.STAGE_PLANS.clear()
        orch_mod.STAGE_PLANS.update(original_plans)

    assert "STAGE_BASE_EXCEPTION" in out, out[-2000:]
    assert "SystemExit" in out
    assert "camera capture died" in out
    # The no-hang contract still holds.
    assert "WorkerDiedError" in result.stages[0].failure_reason


def test_plan_stage1_lifts_the_object_before_carrying_it():
    """The object must leave the counter before the base drives anywhere.

    Stage 1 went close -> carry with no lift, so the robot shut its jaws on
    an object still resting on the counter and dragged it. Measured,
    allobj_6, spoon2: a close passing every hold check
    (gripper_position_rad 0.1227, object_follows_ee True) left object z at
    0.7555 -> 0.7560 across the whole episode, and the carry trace shows
    the jaws staying SHUT while the object fell 1.7 m behind.
    """
    world = MockWorld(seed=3, head_placement="a")
    mem = ParamMemory()
    runner = SelfCorrectingSkill(world, mem, RetryPolicy(mem))

    order: list[str] = []
    real_lift = world.lift
    real_carry = world.carry_object_to

    def _recording_lift(side, dz, **p):
        order.append(f"lift:{dz:.3f}")
        return real_lift(side, dz, **p)

    def _recording_carry(object_name, x, y, z=None, **p):
        order.append("carry")
        return real_carry(object_name, x, y, z, **p)

    world.lift = _recording_lift
    world.carry_object_to = _recording_carry
    plan_stage1(runner, world)

    assert any(s.startswith("lift:") for s in order), (
        "stage 1 never lifted the object off the counter"
    )
    # The lift must come BEFORE the drive, not after it.
    assert order.index(next(s for s in order if s.startswith("lift:"))) < (
        order.index("carry")
    )
    # dz is derived from the scorer's own threshold plus the arm's
    # allowed shortfall, so a lift landing at the tolerance limit still
    # clears min_lift_m.
    expected = (
        config.THRESHOLDS.min_lift_m
        + config.THRESHOLDS.lift_position_tolerance_m
    )
    assert f"lift:{expected:.3f}" in order, order


def test_lift_reports_failure_when_the_object_does_not_rise():
    """The arm reaching its height is not a lift; the object rising is.

    Measured, allobj_7, spoon2: phase lift ok=True with object_rise_m -0.0
    and lift_first_slip_tick 47, while the EE rose 0.8733 -> 0.9403. The
    arm lifted, the spoon did not, and the pipeline called it a success and
    carried on to the carry.
    """
    import types

    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)

    phases: list[tuple[str, bool, dict]] = []
    world._log_phase = lambda name, ok, **kw: phases.append((name, ok, kw))
    world._held = "spoon2"
    world._pre_lift_baseline = {}
    world._last_grasp_offset = {}
    world._m = dict(world._m or {})
    world._m["vgl"] = types.SimpleNamespace(
        object_follows_end_effector=lambda *a, **kw: False
    )
    world.sim = types.SimpleNamespace(cfg=types.SimpleNamespace(dt=1.0 / 60.0))
    world._tick = lambda: None
    # The object never moves, exactly as measured.
    world.object_position = lambda name: (0.0, 0.0, 0.7556)

    class _Arms:
        def ee_world_poses(self):
            return (((0.0, 0.0, 0.8733), (1.0, 0.0, 0.0, 0.0)),) * 2

        def lift(self, side, dz, **p):
            return True  # the ARM succeeds

    world.arms = _Arms()

    result = world.lift("left", 0.08)

    assert result["arm_lift_ok"] is True, "the arm did reach its target"
    assert result["object_rise_m"] == 0.0
    assert result["ik_ok"] is False, "a lift that moved nothing must fail"
    name, ok, _kw = phases[-1]
    assert name == "lift"
    assert ok is False


# --------------------------------------------------------------------------- #
# 2026-08-21 (DOCTOR.md 4.13): the render pass ran on EVERY physics tick while
# almost nothing read a frame. Measured 86-107 ms of wall time per tick against
# a ~1-2 ms physics step, which is the whole of the owner-reported 80-100 min
# per run. The cadence is now derived from what actually consumes frames.
# --------------------------------------------------------------------------- #


def _world_isaac_module():
    import importlib

    return importlib.import_module("task3_pipeline.world_isaac")


def test_render_stride_matches_the_video_capture_cadence_when_recording():
    """Every render produced must be a frame saved -- no discarded passes.

    `_tick`'s video block saves when `tick % capture_every == 0` with
    `capture_every = round(1 / (dt * VIDEO_FPS))`. The render stride has to
    be that same number, or the video either loses frames (stride too
    coarse) or pays for passes nobody reads (stride too fine, which is what
    it did: VIDEO_FPS=2 at dt=0.005 saves 1 frame per 100 ticks, so 99 of
    every 100 renders were thrown away).
    """
    m = _world_isaac_module()
    dt = 0.005
    capture_every = max(1, round(1.0 / (dt * m.VIDEO_FPS)))
    assert m.render_tick_stride(dt, m.VIDEO_FPS, True) == capture_every


def test_render_stride_decimates_hard_when_not_recording():
    """With no video, nothing reads a frame per tick at all.

    Every camera path pumps `app.update()` itself before reading its
    annotators, so it does not depend on the per-tick render. A slow
    keep-alive is all that is left.
    """
    m = _world_isaac_module()
    stride = m.render_tick_stride(0.005, m.VIDEO_FPS, False, idle_hz=10.0)
    assert stride == 20
    assert stride > 1, "un-decimated rendering is the 80-100 min/run bug"


def test_render_stride_never_returns_zero_or_negative():
    m = _world_isaac_module()
    # A dt coarser than the requested cadence must still render every tick,
    # never 0 (which would be a modulo-by-zero in _tick).
    assert m.render_tick_stride(1.0, 2.0, True) == 1
    # This module deliberately runs without pytest (see its header), so
    # assert the raise by hand rather than with pytest.raises.
    for bad in ((0.0, 2.0, True), (0.005, 0.0, True), (-1.0, 2.0, False)):
        try:
            m.render_tick_stride(*bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_ranked_grasp_requires_camera_confirmation_not_telemetry_alone():
    """A telemetry-only "hold" must not be reported as held.

    2026-08-19 owner directive: `object_follows_ee` / contact force have
    been wrong before, so a grip only counts once a SECOND camera (this
    side's own wrist cam, independent of whatever picked the candidate)
    visually confirms the object is between the pads. `reach_and_grasp_
    ranked` returning "held" is what lets the caller move on to lift and
    transport, so a false positive here carries the object nowhere.

    The gate had no test at all until 2026-08-21 -- it was added, wired,
    and silently relied on.
    """
    import task3_pipeline.world_isaac as world_isaac_mod

    world = world_isaac_mod.IsaacWorld(simulation_app=None)
    world._load_ranked_grasp_plan = lambda object_name, side: (
        [_ranked_pair(0, "left", rank=0)],
        _UNMOVED_OBJECT_POSE,
    )
    world.object_position = lambda object_name: _UNMOVED_OBJECT_POSE
    world._append_grasp_attempt_memory = lambda *a, **kw: None
    world.reach = lambda side, object_name, **p: {"position_error_m": 0.05}
    # Telemetry says held, every single time.
    world.grasp = lambda side, object_name, **p: {"object_follows_ee": True}
    # The camera disagrees.
    world.verify_grasp_by_wrist_camera = lambda side, object_name: {
        "verified": False,
        "pixel_frac": 0.0,
        "reason": "object not between pads",
    }

    result = world.reach_and_grasp_ranked("left", "bowl2")

    # The candidate must NOT be reported as the one that worked, and the
    # run must fall through to the hardcoded last resort instead.
    assert result["used_ranked_candidate_id"] is None
    assert result["fell_back_to_hardcoded"] is True
    assert result["attempts"], "the attempt itself should still be recorded"
    assert (
        result["attempts"][0]["grasp"]["camera_verified"]["verified"] is False
    )
