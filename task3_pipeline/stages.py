# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""The four Task 3 stage plans, aligned to the organizer prose rules (the
objective truth -- ``scripts/evaluation/task3/grading.py`` is only a dev
smoke-test, see ``config.py``).

A stage plan is a short sequence of self-correcting skill calls that drives the
world into a state the scorer accepts. The plans encode the *strategy*
decisions from the analysis:

* Stage 1 -- per object (plate, cup, bowl+beans, spoon -- NO tray): navigate
  to a reach-safe kitchen stance, reach, grasp, then carry/place at its
  assigned seat (see ``seats.py``).
* Stage 2 -- scoop with a 30-45 deg entry + deep drag, present smoothly,
  then hold 3 s.
* Stage 3 -- pour the beans back into the recovery sphere low and
  slow (a ratio game).
* Stage 4 -- honest grasp + place of each utensil into the marked sink region
  (navigate -> grasp -> carry/place -> verify). A controlled carry/place is
  legal and sufficient (the scorer needs no cage), but it follows a real,
  verified grasp -- not a physics exploit.

Each returns a StageResult. The orchestrator chains them and sums the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from task3_pipeline import config
from task3_pipeline.outcomes import SkillReport
from task3_pipeline.seats import (
    assigned_seats,
    dining_approach_xy,
    object_to_seat,
)
from task3_pipeline.skills import SelfCorrectingSkill


@dataclass
class StageResult:
    stage: int
    score: int
    max_score: int
    reports: list[SkillReport] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    # Set only when the orchestrator caught this stage failing outright
    # (uncaught exception or wall-clock timeout) instead of scoring it
    # normally -- see task3_pipeline.orchestrator._run_stage_isolated.
    failure_reason: str | None = None

    @property
    def completed(self) -> bool:
        # "completed" for the chain FSM = the stage was attempted and scored
        # at least partial credit. Ranking rewards reaching higher stages, so
        # we always progress regardless (see orchestrator).
        return self.score > 0

    @property
    def fully_completed(self) -> bool:
        # REV12 T3: distinct from `completed` -- every object/attempt this
        # stage could score was actually scored, not just partial credit.
        # A caught exception/timeout path (failure_reason set) always has
        # score=0, so this is correctly False there too.
        return self.max_score > 0 and self.score == self.max_score


# reward helpers for the memory ranking (partial credit in [0,1])
def _r_lower_better(metrics, key, scale):  # e.g. gripper rad, error
    return max(0.0, 1.0 - float(metrics.get(key, scale)) / scale)


def _run(runner, skill, invoke, **kw):
    # 2026-08-14: every skill invocation in both stage plans funnels through
    # here, which makes it the one place a cooperative wall-clock abort
    # covers the whole stage. See config.check_stage_deadline for why the
    # signal.alarm and Thread.join mechanisms both fail to bound a stage in
    # this runtime. No-op unless the caller set a deadline on the world.
    config.check_stage_deadline(getattr(runner, "world", None), f"skill={skill}")
    return runner.run(skill, invoke, **kw)


def _arm_side(world, object_name: str) -> str:
    """REV12 T6: feasibility-based arm-side selection
    (world._select_arm_side, T3) instead of a hardcoded literal. Falls
    back to "right" -- the historical, unchanged default -- for any
    world without the method (MockWorld)."""
    select = getattr(world, "_select_arm_side", None)
    return select(object_name) if select is not None else "right"


# Objects an 85mm parallel jaw cannot reliably pinch single-handed -- the
# standing project finding (the organizers' own recorded demonstrations
# lift the tray and the bowl bimanually, never single-handed; see
# task3_pipeline/config.py's provenance notes) plus the direct 2026-08-16
# measurement that single-arm robotiq cup grasping stalls on the same
# small servo margin through four independent, unrelated fixes while the
# bimanual port (world.reach_bimanual/grasp_bimanual/lift_bimanual,
# world_isaac.py) cleared it cleanly, n=3, real navigation, cup_lift_m
# 0.316. Only "cup" is verified so far -- bowl2/plate2 are the standing
# hypothesis, not yet measured through this path.
BIMANUAL_OBJECTS = frozenset({"cup", "bowl2", "plate2"})


def plan_stage1(runner: SelfCorrectingSkill, world) -> StageResult:
    """Per-object pick -> carry -> place at the assigned seats.

    NO tray, NO single dining-drop point: the organizer rules require the 4
    real objects (config.STAGE1_OBJECTS) to individually end up at 3 of 6
    randomly-assigned seats (seats.py). Seats + the object->seat mapping are
    computed once per stage attempt (not hardcoded per-episode values).
    """
    reports = []
    seats = assigned_seats(seed=getattr(world, "seed", None))
    mapping = object_to_seat(list(config.STAGE1_OBJECTS), seats)

    # 2026-08-14: same restriction stage 4 has had since REVIEW #9, for the
    # same structural reason -- see plan_stage4's comment above `objects`.
    # Stage 1 loops all 4 objects with a full retry budget each, so once ONE
    # object is unreachable the stage cannot finish inside any wall-clock
    # budget, and `run_one` never reaches its `result.as_json()` print --
    # the episode produces NO score at all rather than a low one. This bit
    # actually cost the 2026-08-14 submission runs: v4 ran 60 min and v5 45
    # min, both killed with no result, because the stage never returned.
    # The seat mapping above is deliberately still computed over ALL of
    # STAGE1_OBJECTS, so seat assignment is bit-identical and the scorer is
    # untouched (score_stage(1) still scores 4 objects, max_score stays 4);
    # this only bounds the WORK, exactly like --stage4-objects.
    objects = getattr(world, "stage1_objects", None) or config.STAGE1_OBJECTS

    for obj in objects:
        seat = mapping[obj]
        side = _arm_side(world, obj)

        if obj in BIMANUAL_OBJECTS and hasattr(world, "reach_bimanual"):
            # world.reach_bimanual() owns its own navigation to the proven
            # stance internally (see its docstring) -- no separate
            # "navigate" step here, unlike the single-arm path below, since
            # driving to the single-arm-oriented _stance_for() stance first
            # would just be undone by reach_bimanual's own drive right
            # after.
            reports.append(
                _run(
                    runner,
                    "reach_bimanual",
                    lambda p, o=obj: world.reach_bimanual(o, **p),
                    object_name=obj,
                )
            )
            reports.append(
                _run(
                    runner,
                    "grasp_bimanual",
                    lambda p, o=obj: world.grasp_bimanual(o, **p),
                    object_name=obj,
                )
            )
            if getattr(world, "_held", None) == obj:
                reports.append(
                    _run(
                        runner,
                        "lift_bimanual",
                        lambda p: world.lift_bimanual(
                            config.THRESHOLDS.min_lift_m
                            + config.THRESHOLDS.lift_position_tolerance_m,
                            **p,
                        ),
                        object_name=obj,
                    )
                )
            reports.append(
                _run(
                    runner,
                    "cleanup",  # reuse carry mechanism
                    # 2026-08-16: a real GPU run showed the object separate
                    # from a bimanual friction-only pinch a few seconds into
                    # a full-speed (0.3 m/s default) carry -- gripper_
                    # position_rad never released, so this reads as
                    # inertial slip under the drive's acceleration, not a
                    # dropped/opened grip. Slower base motion means less
                    # inertial force on the pinch; {**p} lets a caller's
                    # own override still win.
                    lambda p, o=obj, s=seat: world.carry_object_to(
                        o,
                        *dining_approach_xy(s),
                        **{"max_linear_mps": 0.15, **p},
                    ),
                    object_name=obj,
                )
            )
            continue

        reports.append(
            _run(
                runner,
                "navigate",
                lambda p, o=obj: world.navigate_to(
                    *world._stance_for(
                        world.object_xy(o), p.get("approach_stance", "north")
                    )[0],
                    reach_check_object=o,
                    **p,
                ),
                object_name=obj,
                reward_fn=lambda m: _r_lower_better(
                    m, "terminal_error_m", 0.2
                ),
            )
        )
        # REV12 T6: ranked-candidate grasp (task3_autonomy.grasp_ranking /
        # grasp_reanchor, T4/T5) replaces the old hardcoded-"right"-arm
        # reach()+grasp() pair -- one skill report covering both, since
        # reach_and_grasp_ranked owns its own fall-through across
        # candidates internally (a different retry mechanism than
        # SelfCorrectingSkill's outer retry, which would otherwise retry
        # the whole ranked sequence rather than just one candidate).
        # Default ON (world.use_ranked_grasp); MockWorld and any world
        # without reach_and_grasp_ranked fall back to the byte-identical
        # two-step path below -- reach_and_grasp_ranked's own "no ranked
        # file" fallback already proves that fallback matches this one
        # call-for-call (task3_pipeline/tests/test_pipeline.py).
        if hasattr(world, "reach_and_grasp_ranked") and getattr(
            world, "use_ranked_grasp", True
        ):
            reports.append(
                _run(
                    runner,
                    "reach_and_grasp",
                    lambda p, o=obj, s=side: world.reach_and_grasp_ranked(
                        s, o, **p
                    ),
                    object_name=obj,
                )
            )
        else:
            reports.append(
                _run(
                    runner,
                    "reach",
                    lambda p, o=obj, s=side: world.reach(s, o, **p),
                    object_name=obj,
                )
            )
            reports.append(
                _run(
                    runner,
                    "grasp",
                    lambda p, o=obj, s=side: world.grasp(s, o, **p),
                    object_name=obj,
                    reward_fn=lambda m: _r_lower_better(m, "gripper_rad", 0.8),
                )
            )
        # LIFT IT OFF THE COUNTER BEFORE DRIVING ANYWHERE.
        #
        # This step did not exist. The sequence went straight from the
        # close to the carry, so the robot shut its jaws on an object that
        # was still resting on the counter and then drove away, dragging it
        # across the surface until it slipped out.
        #
        # Measured, allobj_6, spoon2 -- a close that passed every hold
        # check (`gripper_position_rad 0.1227`, `object_follows_ee True`,
        # `gripper_hold_predicate True`): object z was 0.7555 at the close
        # and 0.7560 at the end of the episode. It never left the counter.
        # The carry trace shows the jaws STAYING SHUT the whole drive
        # -- (23000, 0.1134, 0.108), (23500, 0.0692, 0.239),
        # (26500, 0.1216, 1.667) -- while the object fell 1.7 m behind.
        # Nothing opened; the object was never picked up in the first
        # place, and `object_follows_ee` was measuring a 0.31 m drag along
        # the countertop.
        #
        # dz is DERIVED, not chosen: `min_lift_m` is the rise the scorer
        # requires before a lift counts as real, and `lift()`'s own default
        # `position_tolerance_m` is how far short the arm may legitimately
        # stop. Commanding their sum means that even at the tolerance limit
        # the object still clears `min_lift_m`.
        #
        # Guarded on `_held` so a failed grasp does not command a lift on an
        # empty gripper -- MockWorld records that exact case as the "fling"
        # artifact that was the real Stage-4 bug.
        if getattr(world, "_held", None) == obj:
            reports.append(
                _run(
                    runner,
                    "lift",
                    lambda p, s=side: world.lift(
                        s,
                        config.THRESHOLDS.min_lift_m
                        + config.THRESHOLDS.lift_position_tolerance_m,
                        **p,
                    ),
                    object_name=obj,
                )
            )

        # Carry/place near the object's assigned seat. Target
        # dining_approach_xy(seat), not the seat's raw (x, y) -- every seat
        # position sits literally on the dining table's own surface
        # (seats.py), so driving the base there drives it into the table
        # (measured live, see seats.py's comment). carry_object_to holds
        # the object the whole drive and never releases it, so the object
        # still ends up inside config.DINING_AREA (the real score_stage(1)
        # check) without the base ever touching the table.
        reports.append(
            _run(
                runner,
                "cleanup",  # reuse carry mechanism
                lambda p, o=obj, s=seat: world.carry_object_to(
                    o, *dining_approach_xy(s), **p
                ),
                object_name=obj,
            )
        )

    score, mx, details = world.score_stage(1)
    details = dict(details, assigned_seats=[s.seat_id for s in seats])
    return StageResult(1, score, mx, reports, details)


def plan_stage2(runner: SelfCorrectingSkill, world) -> StageResult:
    reports = []
    reports.append(
        _run(
            runner,
            "scoop",
            lambda p: world.scoop("right", **p),
            reward_fn=lambda m: min(
                1.0, float(m.get("beans_on_spoon", 0)) / 6.0
            ),
        )
    )
    # Present to the feed zone and hold >= 3 s smoothly (head-force safe).
    reports.append(
        _run(
            runner,
            "hold",
            lambda p: world.feed_hold(config.FEED_HOLD_SECONDS, **p),
        )
    )
    score, mx, details = world.score_stage(2)
    return StageResult(2, score, mx, reports, details)


def plan_stage3(runner: SelfCorrectingSkill, world) -> StageResult:
    reports = []
    reports.append(
        _run(
            runner,
            "pour",
            lambda p: world.pour("right", *config.BEAN_SPAWN_XY, **p),
            reward_fn=lambda m: float(m.get("ratio", 0.0)),
        )
    )
    score, mx, details = world.score_stage(3)
    return StageResult(3, score, mx, reports, details)


def plan_stage4(runner: SelfCorrectingSkill, world) -> StageResult:
    """Honest grasp + place of each of the 4 utensils into the sink region.

    Per utensil: navigate to a reach-safe stance -> reach (position the arm
    at the object) -> grasp (verified, see outcomes.classify_grasp) ->
    carry/place into the sink region -> verify. A controlled carry/place is
    legal and sufficient (the scorer checks the object ends in-region at
    height, not a held cage) -- but it follows a real, verified grasp
    rather than a physics exploit. Mirrors plan_stage1's exact
    navigate/reach/grasp/carry order (handoff sec 48: this "reach" step was
    missing here until this fix -- grasp() never moves the arm itself).
    """
    reports = []
    # REVIEW #9 (handoff sec 76): which objects to ATTEMPT. score_stage(4)
    # below still scores all 4 (max stays 4) -- this only bounds the work.
    #
    # Why this exists: grading.py's score_stage4_cleanup returns
    # `score=len(passed)` -- ONE point per object in the sink -- so a single
    # object scores and closes GATE B1 (total_score > 0). Meanwhile
    # STAGE_TICK_BUDGETS[4] (460,800 ticks) is 4.0x more work than
    # HARD_JOIN_CEILING_S=3600 s can execute at the measured 0.0316 s/tick
    # (~113,900 ticks), and even a zero-retry pass over all 4 objects is
    # 115,200 ticks = 3,640 s -- already over the ceiling. So the 4-object
    # loop is GUARANTEED to be killed mid-run by TimeoutError no matter how
    # good the physics is, and the CLEANUP_GRID retry search never runs.
    # Restricting the loop to one object converts a structurally impossible
    # stage into a winnable one; it does not change the scorer.
    objects = getattr(world, "stage4_objects", None) or config.STAGE1_OBJECTS
    for obj in objects:

        def _navigate_stance(p, o=obj):
            # R7 T4: cup's grasp stance uses VM B's proven fixed point
            # (config.CUP_GRASP_STANCE_XY, see its own comment) instead of
            # world._stance_for()'s general search, which a live GPU test
            # showed picking an unreliable stance for this same call site.
            # Every other Stage 4 object is unaffected -- this is scoped to
            # "cup" only, not a change to _stance_for() itself.
            if o == "cup":
                return (
                    config.CUP_GRASP_STANCE_XY,
                    config.CUP_GRASP_STANCE_YAW_RAD,
                )
            return world._stance_for(
                world.object_xy(o), p.get("approach_stance", "north")
            )

        def _run_navigate(p, o=obj):
            stance_xy, stance_yaw = _navigate_stance(p, o)
            # 2026-08-09 (O1 investigation): world_isaac.py's
            # navigate_to_avoiding_island's docstring -- the same
            # island-crossing bug found in _push_object_to's stance
            # navigate also hit this general "navigate to object" step
            # (spoon2_run1.log: stalled at terminal_error_m 1.5 on the
            # very first call, no push involved yet). "cup" is unaffected
            # either way (its hand-fit CUP_GRASP_STANCE_XY never crosses
            # the island); every other object gets the same safety net.
            return world.navigate_to_avoiding_island(
                *stance_xy, stance_yaw, reach_check_object=o, **p
            )

        reports.append(
            _run(
                runner,
                "navigate",
                _run_navigate,
                object_name=obj,
                reward_fn=lambda m: _r_lower_better(
                    m, "terminal_error_m", 0.2
                ),
            )
        )
        # REVIEW #10 (handoff sec 86): reach()'s justification above (a
        # Cartesian IK jump straight from reset, sec 48) was made false by
        # sec 72's own fix -- _push_object_to now re-stances, navigates,
        # rotates, releases the gripper, and does its own push_approach
        # pre-pose itself (world_isaac.py ~:1253-1294). reach() reads
        # nothing _push_object_to uses; it only puts the gripper DOWN ONTO
        # the object (a grasp pre-pose), which for a push is a pure contact
        # hazard with no downstream consumer -- the deterministic cause of
        # sec 85's plate2 floor (and sec 78/82's cup fling). Removed rather
        # than tuned: Iron Rule 8 closed re-tuning grasp/descend geometry
        # constants, and this path shouldn't execute for a push at all.
        # C0 diagnostic (handoff sec 49, R2.2): --skip-grasp lets this call
        # be bypassed entirely so cleanup's push runs straight after
        # navigate, isolating whether the failing grasp itself (not the
        # push) is what flings objects off the counter before push ever
        # gets a chance.
        if not getattr(world, "skip_grasp", False):
            # REV12 T6: feasibility-based side (world._select_arm_side,
            # T3) instead of the hardcoded "right" literal. Deliberately
            # NOT reach_and_grasp_ranked here -- that always calls
            # reach() internally, and REVIEW #10 (above) removed reach()
            # from this exact call site on purpose (a Cartesian
            # pre-grasp descent onto the object is a documented contact
            # hazard for _push_object_to's push strategy, the
            # deterministic cause of the plate2-floor/cup-fling
            # incidents) -- reintroducing it here would be exactly the
            # regression the hard rules forbid ("DO NOT regress
            # --carry-to-sink or the cup path").
            reports.append(
                _run(
                    runner,
                    "grasp",
                    lambda p, o=obj: world.grasp(_arm_side(world, o), o, **p),
                    object_name=obj,
                    reward_fn=lambda m: _r_lower_better(m, "gripper_rad", 0.8),
                )
            )
        def _cleanup(p, o=obj):
            # 2026-08-09 (O1 investigation): CLEANUP_GRID declares "method"
            # first/outermost (config.py's own comment), so within
            # RETRY_BUDGET=4 attempts the grid NEVER reaches
            # method="grasp_place" -- every attempt is "base_carry" (push),
            # even on a run where grasp() was NOT skipped and genuinely
            # succeeded. That grid ordering was a deliberate, evidenced
            # choice for the --skip-grasp default (grasp_place is a
            # guaranteed no-op there), but it silently wastes a real,
            # already-GPU-verified mechanism (1d2f619: hand-checked
            # continuous object movement through close/lift/hold/carry,
            # landed 10-12cm inside SINK_BOUNDS) whenever grasp() DOES run
            # and DOES succeed. `world._held == o` is the same real
            # follows-the-end-effector check grasp() itself uses (not a
            # bare contact flag) -- when it's true, use the mechanism
            # that's proven to work for a genuine hold instead of the grid's
            # push default; otherwise behavior is byte-for-byte unchanged.
            if getattr(world, "_held", None) == o:
                p = dict(p)
                p["method"] = "grasp_place"
            # 2026-08-09 (O1 investigation, owner requested a working
            # demo): CLEANUP_GRID alternates approach_stance between
            # "north"/"east" every retry. This session's own GPU evidence
            # (spoon2 push-phase logs) shows "north" stances for objects
            # near the west counter land right at/inside the room's real
            # west wall (WEST_WALL_BBOX), while "east" stances are the
            # ONLY ones that have ever reached push_contact or converged
            # push_approach cleanly this session. Force "east" for
            # cleanup specifically -- a scoped override of one grid
            # dimension, not a change to the grid or any other skill --
            # so every retry attempt spends its budget on the direction
            # that has actually shown real progress, not half of them on
            # one this session's own data says is worse.
            #
            # CORRECTION 2026-08-14: the diagnosis above was wrong about
            # WHY. Those "north" stances were not near a wall --
            # `WEST_WALL_BBOX` was wrong by 1.64 m and there is no wall
            # there (see its comment in `task3_autonomy/navigation.py`).
            # They were inside the kitchen COUNTER, because
            # `_rotate_to_clear_island` had no clearing candidate at all
            # and silently returned its un-cleared nominal. The
            # observation was real; the explanation was not. With that
            # fixed, "north" and "east" now sweep to nearly the same
            # stance west of the counter, so this override is close to a
            # no-op. Left in place -- it costs nothing and this is not
            # the session to re-open the cleanup grid -- but do not carry
            # the west-wall reasoning forward.
            #
            # CORRECTION 2026-08-21: "it costs nothing" was wrong, and the
            # cost is Stage 4's entire score. Every piece of evidence this
            # override ever cited is about the PUSH path -- push_contact,
            # push_approach, "spoon2 push-phase logs". It was written when
            # cleanup was always `base_carry`. The `method="grasp_place"`
            # branch directly above landed later and independently, and
            # forcing "east" onto THAT path hits the one combination known
            # to fail: `carry_object_to` rejects `grasp_place` from an east
            # stance outright ("grasp_place from east flings object",
            # task3_pipeline/world.py) and returns without moving the
            # object. So every cleanup of a genuinely HELD object -- the
            # case the grasp_place branch exists to serve -- scored zero.
            # Measured on MockWorld seed 7: stage 4 = 0/4, objects never
            # moved, and the episode capped at 11/16 = 0.688 against a 0.70
            # gate.
            #
            # Scoped to the push path, which is the only path its evidence
            # was ever about. Two separately reasonable changes; the defect
            # was in their interaction.
            p = dict(p)
            if p.get("method") != "grasp_place":
                p["approach_stance"] = "east"
            return world.carry_object_to(
                o,
                *config.nearest_sink_aim_point(world.object_xy(o)),
                config.SINK_TABLETOP_Z,
                **p,
            )

        # 2026-08-09 (O1 investigation, owner requested a working demo):
        # temporary, SCOPED retry-budget bump for "cleanup" only --
        # config.RETRY_BUDGET=4 is a single shared attribute on the one
        # RetryPolicy instance `runner` uses for every skill this whole
        # episode, so mutating it here and restoring it right after
        # affects only this one call, not navigate/grasp/other objects'
        # cleanup (verified: `SelfCorrectingSkill.run` reads
        # `self.policy.budget` fresh via `range(self.policy.budget + 1)`
        # each call, single-threaded, no concurrent readers to race).
        # GPU-confirmed this session (joint-thrash guard, world_isaac.py):
        # retries are now safe (no cumulative object damage across
        # attempts), so more attempts is a real, low-risk lever it wasn't
        # before. Doubling from 4 to 8, not an unbounded change.
        cleanup_budget_bump = 8
        original_budget = runner.policy.budget
        runner.policy.budget = cleanup_budget_bump
        try:
            reports.append(
                _run(
                    runner,
                    "cleanup",
                    _cleanup,
                    object_name=obj,
                )
            )
        finally:
            runner.policy.budget = original_budget
    score, mx, details = world.score_stage(4)
    return StageResult(4, score, mx, reports, details)


STAGE_PLANS = {1: plan_stage1, 2: plan_stage2, 3: plan_stage3, 4: plan_stage4}


def official_spec_ready(stage: int, details: dict) -> bool:
    """Would this stage's score count under the REAL organizer rules, not
    just ``grading.py``'s dev-smoke-test scorer (config.py's own docstring:
    "NOT the official scorer")? P0.9 (handoff sec 16.10(D)/17.4 #5/17.6).

    Every stage's numeric ``score`` is the *development_score* -- what the
    local scorer says -- regardless of this flag; this only marks whether
    that number is also defensible as a competition claim.
    """
    if stage == 1:
        # score_stage1_table_setup only checks "is this object roughly in
        # the dining rectangle" (config.py sec: DINING_AREA is a coarse
        # fallback) -- it never reads the object's actual assigned seat
        # identity from seats.py, so a dev-scorer pass here is not yet a
        # real-rules pass (handoff sec 16.2 row 7).
        return False
    if stage == 2:
        # Safety is a HARD FAIL (sec 1): no point may be banked without a
        # measured peak head-contact force under HEAD_MAX_FORCE_N.
        peak = details.get("peak_head_force_n")
        return peak is not None and peak <= config.HEAD_MAX_FORCE_N
    if stage == 3:
        # CORRECTION (handoff.md sec 18.4): BEAN_RECOVERY_CENTER sits on the
        # beans' OWN start XY (config.BEAN_SPAWN_XY) and plan_stage3 pours
        # back at that same point -- grep for "recycl|trash|bin_" over
        # grading.py/config.py returns zero hits, so there is no evidence
        # the scored sphere is the "recycling bin" sec 1's prose describes.
        # A high dev-scorer ratio may just mean the beans never moved.
        # Forced False until the organizer prose is re-read to resolve
        # which of {prose wrong, sphere wrong} is true.
        return False
    # Stage 4 (sink bounds + tabletop z) matches the real prose-rule
    # geometry per config.py's own sourcing -- no known lenient-scorer gap.
    return True
