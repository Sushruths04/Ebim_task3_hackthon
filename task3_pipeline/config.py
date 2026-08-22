# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Static configuration for the Task 3 pipeline.

IMPORTANT: ``scripts/evaluation/task3/grading.py`` is the organizers'
DEV-TIME smoke-test helper, NOT the official scorer -- their own README says
so verbatim. It scores a lenient dining *rectangle* and (in its current form)
wrongly includes ``simple_tray``. **Do not treat it as the definition of
done.** The objective truth is the organizer prose rules
(https://ebim-benchmark.github.io/competition.html): real Stage 1 = carry 4
objects (plate, cup, bowl+beans, spoon -- NO tray) from the kitchen to 3 of 6
seats, randomly assigned per episode (see ``seats.py``), not a single drop
point inside a dining rectangle.

Sink/bean geometry below (Stage 3 recovery, Stage 4 sink) DOES match the real
rules and is kept as the source of truth for those stages. ``DINING_AREA`` is
kept ONLY as a coarse fallback / CPU smoke-test reference (cheap sanity check
that an object is "roughly in the dining room"), not as the real Stage-1
target -- the real target is the 3 assigned seat coordinates from
``seats.py``.

The parameter GRIDS are the search space the self-correction loop explores
instead of a human hand-editing one constant at a time. Keep them small and
bounded: the loop tries them in order, best-known-first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# World geometry
# --------------------------------------------------------------------------- #

# Stage 1 (dev smoke-test ONLY, see module docstring): coarse "is this object
# roughly in the dining room" rectangle. The REAL Stage-1 target is the 3
# assigned seats resolved by ``seats.py`` (per organizer prose rules).
DINING_AREA = dict(center_x=-2.85, center_y=1.9, scale_x=5.9, scale_y=3.4)
KITCHEN_AREA = dict(center_x=-4.2, center_y=-1.8, scale_x=3.2, scale_y=4.1)
# The 4 REAL Stage-1 objects per the organizer prose rules -- NO tray. Objects
# start stacked on a plate in the kitchen and are carried individually (or as
# an optimization, riding the plate) to their assigned seat (see seats.py).
#
# ORDER, 2026-08-14. The set is unchanged and every object is still
# attempted; only the sequence moved, and it moved for a measured reason.
# `plate2` was first, and a plate lying flat is the one object here a
# two-finger parallel jaw has no purchase on -- there is no clearance under
# its rim to close on. Runs 3-8 all show the same thing: the jaws push it
# rather than grasp it, it slides ~14 cm across the counter and ends on the
# floor, and its three retries plus the push fallback consume most of the
# stage budget BEFORE any other object is tried even once. Attempting the
# objects the gripper can actually close on first means a failure on the
# hardest one no longer costs the other three their attempt.
#
# Corroboration that this is the objects' nature and not our bug: the
# organisers' own recorded demonstrations
# (`bburdiek/task3_feeding_cutlery_tray_pickup`,
# `bburdiek/task3_feeding_bowl_spoon_pickup`) pick up the TRAY and the BOWL.
# Neither dataset picks up the plate.
#
# REFINED 2026-08-15 on the first successful close of the project. `spoon2`
# is now first, and the reason is mechanical before it is empirical: an
# 85 mm two-finger parallel jaw needs two opposing surfaces close enough
# together to pinch, and the spoon's narrow STEM is the only feature in this
# scene that is one. A cup rim, a bowl rim and a flat plate all present a
# thin wall the pads either miss or slide off -- which is why the
# organisers' own demonstrations lift the tray and the bowl BIMANUALLY
# rather than pinching either single-handed.
#
# The measurement agrees (run 19, one 3600 s stage): spoon2 was held on its
# FIRST live_er_grasp attempt -- `close ok=True`,
# `gripper_position_rad 0.0532` above the 0.05 hold band,
# `object_follows_ee True`, and the object then travelled 0.43 m with the
# gripper. cup took three attempts and never held; bowl2 one, and never
# held. Attempts cost ~12 minutes each including navigation, so the stage
# timed out at 55 minutes with its one working grasp still uncarried.
# Ordering by what the gripper can physically close on is worth more than
# any other change available here. (n=1 on the empirical half; the
# mechanical argument does not depend on it.)
STAGE1_OBJECTS = ("spoon2", "cup", "bowl2", "plate2")
# Smoke-test-only fallback drop point well inside the dining rectangle (kept
# for the dev scorer path; the real path targets per-object seat coordinates).
DINING_DROP_XY = (-2.85, 1.9)

# Stage 3: beans score by ratio inside this sphere (which sits at the bowl).
BEAN_RECOVERY_CENTER = (-3.9436692, -1.9169676, 0.8598584)
BEAN_RECOVERY_RADIUS = 0.2
BEAN_SPAWN_XY = (-3.94, -1.92)
STAGE3_RATIO_FOR_2PTS = 0.8
STAGE3_RATIO_FOR_3PTS = 0.9

# Stage 4: an object scores if its XY footprint overlaps the sink rect AND
# z >= tabletop_z. NOTE: no grasp / lift / hold is required by the scorer.
SINK_BOUNDS = dict(
    x_min=-4.245322, y_min=-2.412793, x_max=-3.805322, y_max=-2.042793
)
SINK_TABLETOP_Z = 0.74699
SINK_CENTER_XY = (
    (SINK_BOUNDS["x_min"] + SINK_BOUNDS["x_max"]) / 2.0,
    (SINK_BOUNDS["y_min"] + SINK_BOUNDS["y_max"]) / 2.0,
)  # (-4.0253, -2.2278)


# R7 T4 (plans/SYNC.md 2026-08-04 ~21:17 UTC): plan_stage4's own "navigate"
# step computes the cup-grasp stance via `world._stance_for()` -- the same
# general-purpose curobo/fallback search that this whole project's push
# work spent a full session finding unreliable. A live GPU test
# (r7t4_grasp_test) confirmed the same instability here: the search parked
# the base at [-4.877, -0.939] facing almost directly away from the cup,
# `recenter`'s reach() failing 800/800 IK ticks at a 1.45m target norm.
# VM B's `verify_grasp_lift.py` (rev 8, 2/2 real carry-to-sink scores)
# never uses this search at all -- it drives to a hand-measured, verified
# fixed point instead (`STANCE = (-3.32, -1.72)`,
# `probe_room_geometry_near_point.py`-verified clearance, "base front
# 0.05m off the island east face"). Reused here verbatim, not re-derived --
# the same real, evidenced value VM B's own GPU runs already validated.
CUP_GRASP_STANCE_XY = (-3.32, -1.72)
CUP_GRASP_STANCE_YAW_RAD = math.pi  # faces west, toward the cup


def scores_in_sink(x: float, y: float, z: float) -> bool:
    """Mirror of ``grading.py``'s ``score_stage4_cleanup`` predicate, for use
    by the pipeline's OWN internal success checks.

    REVIEW #9 (handoff sec 76): ``_push_object_to`` used to judge its own
    result with hand-picked tolerances -- ``hypot(final_xy - target_xy) <=
    0.5`` and ``final_z >= target_z - 0.05`` -- while the real scorer
    requires the object's XY footprint to overlap ``SINK_BOUNDS`` (a
    0.44 x 0.37 m rect, half-diagonal only ~0.29 m) and ``z >=
    SINK_TABLETOP_Z`` with NO slack at all. Both hand-picked tolerances
    were looser than the thing they claimed to measure, so the pipeline
    could log ``scored: True`` on an object the official scorer rejects --
    the same "every gate is looser than what it measures" failure this
    project has now hit four times (handoff sec 4.64/18.1/D1).

    Deliberately CONSERVATIVE: the official check is an AABB overlap, this
    is point containment. Point-inside implies AABB-overlaps, so anything
    this returns True for the official scorer also accepts -- never the
    other way round. Pure arithmetic, CPU-testable, no Isaac import.
    """
    return (
        SINK_BOUNDS["x_min"] <= x <= SINK_BOUNDS["x_max"]
        and SINK_BOUNDS["y_min"] <= y <= SINK_BOUNDS["y_max"]
        and z >= SINK_TABLETOP_Z
    )


# T3 (plans/LOOP_PROMPT_VM_A_REV4.md): every Stage-4 push in this project's
# history has aimed at SINK_CENTER_XY (stages.py's old call site), which is
# 30-40% farther than the scorer requires -- point-in-rect only needs the
# object's NEAREST point inside SINK_BOUNDS, not the geometric middle. A
# smaller margin means less travel, less time straining past the arm's
# ~0.855m reach ceiling (handoff sec 105/106), and no downside: point-inside-
# the-shrunk-box still implies point-inside-SINK_BOUNDS.
#
# rev 12 T1/T6 follow-up (plans/SYNC.md 2026-08-06/07): 2/2 real
# --carry-to-sink episodes that reached a real hold+carry attempt this
# session both missed SINK_BOUNDS, SHORT in -y, on the SAME edge
# (y_max=-2.042793): 7.7cm short (SYNC 64), then 1.9cm short (SYNC 68).
# Root cause measured, not guessed: the carry's own base drive
# (verify_grasp_lift.py's target_base_xy) fell short of ITS OWN commanded
# target by 2.8cm in the first case, and the held object's rigid-offset
# assumption drifts further on top of that. 0.05m of margin is smaller
# than the worst measured shortfall, i.e. it could not have absorbed
# either miss. Raised to comfortably clear both with buffer to spare;
# still well inside the box (usable clamped region shrinks from 0.34x0.27m
# to 0.24x0.17m, no meaningful increase to arm reach demand).
SINK_AIM_MARGIN_M = 0.12


def nearest_sink_aim_point(
    object_xy: tuple[float, float],
) -> tuple[float, float]:
    """The closest point to ``object_xy`` that still scores, with a safety
    margin so the push doesn't have to land exactly on the rect edge.

    Clamps ``object_xy`` into ``SINK_BOUNDS`` shrunk by
    ``SINK_AIM_MARGIN_M`` on every side, rather than aiming at
    ``SINK_CENTER_XY`` for every object regardless of where it starts. Pure
    arithmetic, CPU-testable, no Isaac import -- mirrors ``scores_in_sink``'s
    own conservatism (shrinking the box, not growing it, means anything this
    aims at is still comfortably inside the real ``SINK_BOUNDS``, never on
    or outside its edge).
    """
    x_min = SINK_BOUNDS["x_min"] + SINK_AIM_MARGIN_M
    x_max = SINK_BOUNDS["x_max"] - SINK_AIM_MARGIN_M
    y_min = SINK_BOUNDS["y_min"] + SINK_AIM_MARGIN_M
    y_max = SINK_BOUNDS["y_max"] - SINK_AIM_MARGIN_M
    x, y = object_xy
    return (min(max(x, x_min), x_max), min(max(y, y_min), y_max))


# Stage 2: feeding requires smooth motion + >= 3 s hold with beans on the
# spoon in the feed zone (a small box in front of the head placement).
FEED_HOLD_SECONDS = 3.0
FEED_ZONE_RADIUS_M = 0.15
FEED_SMOOTH_MAX_STEP_M = 0.05  # movement_is_smooth() gate

# Sec 1: "Safety = HARD FAIL: peak head/face force (ISO/TS 15066) + watchdog."
# ISO/TS 15066:2016 Annex A Table A.2's face region carries the lowest
# transient contact-force ceiling on the whole body: 65 N (quasi-static/
# sustained-clamping limits are stricter, roughly 40-65% of the transient
# value, but Stage 2's spoon-to-head approach is a brief proximity event,
# not a clamp). P0.9 (handoff sec 16.10(D)/17.6): no Stage-2 point may be
# banked without a measured peak head-contact force under this ceiling.
HEAD_MAX_FORCE_N = 65.0

# Known FSM waypoints (from task3_master_plan.md / repo).
ROBOT_SPAWN = (-4.6, 2.7, -1.5708)  # x, y, yaw
CUP_SPAWN_XY = (-4.185, -1.753)


# --------------------------------------------------------------------------- #
# Verifier thresholds -- how the auto-verifier decides SUCCESS vs a failure
# mode from measured numbers you already log today.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VerifierThresholds:
    # A firm rim cage closes to ~0.076 rad (verify_grasp_lift.py 10/10).
    # Above this the gripper only caught the body -> WEAK_GRASP.
    grasp_cage_max_rad: float = 0.20
    # Reach counts as landed only within this Cartesian error.
    reach_tolerance_m: float = 0.05
    # Object must rise at least this much to count a lift as real.
    min_lift_m: float = 0.05
    # How far short of its commanded lift the arm may legitimately stop.
    # Lives here, next to min_lift_m, because the two are only meaningful
    # together: the commanded dz must be their SUM, so that an arm stopping
    # at the tolerance limit still clears min_lift_m and the lift counts.
    lift_position_tolerance_m: float = 0.03
    # If object z falls more than this after a hold started -> SLIP.
    slip_drop_m: float = 0.03
    # Navigation terminal tolerance.
    nav_tolerance_m: float = 0.05
    # ACTIVE_BRIEF.md sec 3.5/T3: a navigate that stops short of its target
    # but still leaves the object within this distance of the achieved base
    # pose is a real success, not NAV_SHORT -- classify_navigate() checks
    # this against "object_dist_m" when the caller supplies it. Below the
    # ~0.865-0.87 m stance reach budget with a small margin.
    nav_reach_envelope_m: float = 0.8
    # Object-follows-EE tolerance for the honest grasp check: even with the
    # cage angle closed and contact reported, the grasp only counts as a real
    # hold if the object is within this distance of the end-effector (closes
    # the recurring "gripper closed on empty air" bug -- see outcomes.py).
    GRASP_HELD_MAX_DIST_M: float = 0.08


THRESHOLDS = VerifierThresholds()


# --------------------------------------------------------------------------- #
# Bounded parameter search grids (the "18 manual runs", automated).
# Ordered best-known-first; the RetryPolicy walks them on failure.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParamGrid:
    """One skill's tunable knobs and the discrete values worth trying."""

    name: str
    grid: dict[str, tuple] = field(default_factory=dict)


# Grasp / cleanup reachability grid. The Stage-4 root cause was reachability
# (arm can't reach the Y-offset from the east stance), so STANCE is first.
GRASP_GRID = ParamGrid(
    name="grasp",
    grid={
        "approach_stance": ("north", "east"),  # stance first: fixes reach
        "grasp_y_offset": (0.0, 0.04, 0.06),
        "grasp_height_above_origin_m": (0.068, 0.10),
        "base_hold_kp": (4.0, 8.0, 12.0),
    },
)

# Stage-4 cleanup can bypass grasping entirely (scorer needs no cage).
#
# T4 (LOOP_PROMPT_VM_A_REV4.md): confirmed by reading skills.py/policy.py,
# not assumed -- SelfCorrectingSkill.run() calls itertools.product(*grid.
# values()) (declaration order) with NO diagnosis-driven reordering for
# this skill ("cleanup" is absent from outcomes.CLASSIFIERS, so every
# failure classifies as UNSCORED, which _OUTCOME_PRIORITY_KEY has no entry
# for). The OLD key order (method, approach_stance, offset, offset) meant
# the first RETRY_BUDGET=4 real attempts were ALL (method="base_carry",
# approach_stance="north") -- only the two contact-geometry offsets ever
# varied; "east" was never tried inside the budget at all (verified: real
# plan() output pasted in handoff sec 108).
#
# Two real, evidenced fixes, not just a reorder for its own sake:
# 1. "controlled_slide" dropped -- carry_object_to's own comment
#    (world_isaac.py) says it and "base_carry" route to IDENTICAL code
#    (one real push mechanism); keeping both as distinct grid values wasted
#    a retry slot on a byte-for-byte rerun disguised as a different
#    attempt. "grasp_place" is kept (a real, different code path) but
#    stays last/outermost: it requires self._held == object_name and this
#    project's own scored command runs Stage 4 with --skip-grasp (see the
#    2026-08-02 proof bundle's repro command), under which grasp_place is
#    a GUARANTEED no-op every time -- not worth spending one of only 4
#    retries on when it can't possibly succeed in that mode.
# 2. approach_stance moved to vary FASTEST (last key = itertools.product's
#    innermost loop) so both stances get tried inside the first 2 attempts,
#    not zero. Caveat, evidenced this same task (handoff sec 108): in the
#    FALLBACK path (navigation.stance_for(), used ~59% of the time per T2),
#    "north" and "east" collapse to nearly the SAME rotated position for
#    cup/spoon2/plate2 once island-avoidance is applied -- so this reorder
#    mainly helps when curobo's own search is live, not the fallback. Kept
#    in the grid rather than dropped: it is real code, still occasionally
#    distinct (bowl2), and removing a whole axis is a bigger, unverified
#    change than reordering one.
CLEANUP_GRID = ParamGrid(
    name="cleanup",
    grid={
        "method": ("base_carry", "grasp_place"),
        # C0 (world_isaac.py _push_object_to): where behind the object the
        # gripper contacts it, and at what height offset from the object's
        # own origin -- left for the self-correction loop to search rather
        # than hand-guessed, per this project's own "grid search replaces
        # human hand-tuning" rule.
        "push_behind_offset_m": (0.06, 0.09),
        "push_contact_height_offset_m": (0.0, -0.02),
        "approach_stance": ("north", "east"),
    },
)

SCOOP_GRID = ParamGrid(
    name="scoop",
    grid={
        "entry_pitch_deg": (35.0, 45.0, 30.0),
        "drag_depth_m": (0.03, 0.05),
        "scoop_speed": ("slow", "medium"),
    },
)

POUR_GRID = ParamGrid(
    name="pour",
    grid={
        "pour_height_m": (0.05, 0.03, 0.08),
        "tilt_rate": ("slow", "medium"),
    },
)

REACH_GRID = ParamGrid(
    name="reach",
    grid={"approach_stance": ("north", "east")},
)

NAVIGATE_GRID = ParamGrid(
    name="navigate",
    grid={
        # ACTIVE_BRIEF.md T1: navigate targets a computed stance, not the
        # raw object xy -- stance first (north-first, matching the other
        # grids), so a bad approach direction is fixed before scalar-tuning
        # speed.
        "approach_stance": ("north", "east"),
        "max_linear_mps": (0.5, 0.3),
    },
)

GRIDS: dict[str, ParamGrid] = {
    g.name: g
    for g in (
        GRASP_GRID,
        CLEANUP_GRID,
        SCOOP_GRID,
        POUR_GRID,
        REACH_GRID,
        NAVIGATE_GRID,
    )
}

# Per-skill retry budget for the fast loop (partial points beat a hang).
RETRY_BUDGET = 4

# Where the persistent parameter/failure memory lives.
DEFAULT_MEMORY_PATH = "outputs/task3_pipeline/param_memory.json"

# --------------------------------------------------------------------------- #
# Orchestrator fault isolation (see orchestrator.py)
# --------------------------------------------------------------------------- #

# Every stage is worth 4 points; used as the score/max recorded for a stage
# that the orchestrator catches failing outright (exception or timeout).
STAGE_MAX_SCORE = 4

# CORRECTION (handoff.md sec 18.1, W0.5): this was wall-clock SECONDS
# compared directly against skill budgets denominated in SIM ticks
# (world_isaac.py navigate_to/_rotate_to/carry_object_to are all
# `int(budget_s / dt)`, dt=0.005). At this repo's own measured tick rate
# (0.051-0.116 s wall/tick, handoff sec 18.1) Stage 1 needs ~1,877-4,258s
# even executing perfectly -- 3-7x over this number. Four sessions'
# timeouts (handoff sec 4.60/4.63/4.64/4.65) were predicted by this
# arithmetic alone and carried zero information about the physics.
# KEPT ONLY as an explicit override escape hatch (run_task3.py
# --stage-timeout-s, used by the single-stage isolating experiments) --
# it is NOT the default enforcement value any more. Raising this number
# and calling it fixed would still be a wrong unit; see
# stage_wallclock_ceiling_s() below for the actual default.
STAGE_WALLCLOCK_BUDGET_S = 600.0

# --------------------------------------------------------------------------- #
# Stage TICK budgets (handoff.md sec 18.1c/18.6 W0.5) -- the real budget unit.
# Every skill's own internal budget is in SIM ticks, so the orchestrator's
# stage budget must be too. The wall-clock number actually passed to
# threading.Thread.join() (orchestrator._run_stage_isolated, a real
# infrastructure abort only -- Python cannot interrupt mid-tick-loop any
# other way) is DERIVED from this tick budget times a MEASURED s_per_tick,
# never chosen independently -- see stage_wallclock_ceiling_s().
# --------------------------------------------------------------------------- #

# Per-object worst case, all 5 skill calls in stages.py's plan_stage1/
# plan_stage4 object loop hitting their full declared budget_s once
# (handoff sec 18.1's own table): navigate 9,000 + reach's internal
# navigate_to(budget_s=25) 5,000 + reach's rotate 3,000 + reach's arm
# pregrasp+descend >=3,200 + carry 8,000 + release/settle ~600
# = >=28,800 ticks/object.
_STAGE1_4_TICKS_PER_OBJECT = 28_800

# Stage 2 (stages.py plan_stage2 -> world_isaac.py scoop()/feed_hold()):
# reach+grasp(spoon) pickup (25+15+8+8 sim-s) + scoop_enter+scoop_lift
# (5+5 sim-s, world_isaac.py:1027/1037 timeout_s=5.0) + feed_hold's own
# doorway navigate (up to 3 legs x budget_s=45, world_isaac.py:1095) +
# rotate (10s, :1104) + insertion reach (8s, :1135) + hold+recovery
# (3s + 15s, :1151-1152).
_STAGE2_SIM_SECONDS = (
    25.0 + 15.0 + 8.0 + 8.0 + 5.0 + 5.0 + 3 * 45.0 + 10.0 + 8.0 + 3.0 + 15.0
)

# Stage 3 (stages.py plan_stage3 -> world_isaac.py pour()): reach+grasp
# (bowl) pickup (25+15+8+8 sim-s) + the tilt/pour motion itself (15 sim-s,
# conservative -- pour() has no single declared budget_s of its own).
_STAGE3_SIM_SECONDS = 25.0 + 15.0 + 8.0 + 8.0 + 15.0

SIM_DT_S = 0.005  # task3_pipeline/world_isaac.py SimulationCfg(dt=0.005)

# x RETRY_BUDGET: a stage that legitimately needs every retry on every
# object/skill must not be falsely killed by its own tick budget.
STAGE_TICK_BUDGETS: dict[int, int] = {
    1: _STAGE1_4_TICKS_PER_OBJECT * len(STAGE1_OBJECTS) * RETRY_BUDGET,
    2: int(_STAGE2_SIM_SECONDS / SIM_DT_S) * RETRY_BUDGET,
    3: int(_STAGE3_SIM_SECONDS / SIM_DT_S) * RETRY_BUDGET,
    4: _STAGE1_4_TICKS_PER_OBJECT * len(STAGE1_OBJECTS) * RETRY_BUDGET,
}

# sec 21 (2026-07-26): superseded by a real measurement -- the W1 tick
# heartbeat (WORLD_ISAAC_TICK) on a Lightning L4, `--order 4
# --skip-navigation`, gave a clean s_per_tick of 0.025-0.045 across two
# independent runs (mean ~0.029), both BEFORE and AFTER the sec 21
# render-thread-deadlock fix -- confirming render=False bought ~0% speed
# (its value is the deadlock fix alone, not a throughput win; main-thread
# render=True ticks and worker-thread render=False ticks measured the same
# rate). The previous 0.1157 fallback (handoff sec 18.1's proof bundle, an
# undocumented camera setting) was ~4x too pessimistic and inflated every
# derived stage_wallclock_ceiling_s(). Still a fallback, not a per-run
# constant -- pass --measured-s-per-tick with a fresh measurement when one
# is available.
MEASURED_S_PER_TICK_FALLBACK = 0.03


# sec 19b W1.3: before this cap, stage 1/4's derived ceiling resolved to
# 79,972s (22.2h) -- a silently unbounded wall for a genuinely stuck stage,
# which is what turned "the join times out after 10 min" (sec 4.60, a
# recoverable failure) into "the process sits for a day" once the derived
# ceiling replaced the flat 600s budget (sec 18.1's W0.5). This is an
# infrastructure-abort ceiling only; it does not change STAGE_TICK_BUDGETS,
# which remains the real, tick-based budget.
HARD_JOIN_CEILING_S = 3600.0


def stage_wallclock_ceiling_s(
    stage: int, measured_s_per_tick: float | None = None
) -> float:
    """Infrastructure-abort-only wall-clock ceiling for
    ``orchestrator._run_stage_isolated``'s ``thread.join()``.

    This is NOT a budget in its own right -- it exists only because Python
    cannot cooperatively interrupt a stage thread mid tick-loop any other
    way. The real budget is ``STAGE_TICK_BUDGETS[stage]``; this converts it
    to wall seconds via a MEASURED tick rate (never an independently
    chosen wall-clock guess -- that was sec 18.1's root cause), with a 1.5x
    margin so scheduling jitter alone cannot trip it. Capped at
    ``HARD_JOIN_CEILING_S`` so a derived ceiling can never itself become an
    unbounded silent wall (sec 19b).
    """
    rate = (
        measured_s_per_tick
        if measured_s_per_tick is not None
        else MEASURED_S_PER_TICK_FALLBACK
    )
    return min(STAGE_TICK_BUDGETS[stage] * rate * 1.5, HARD_JOIN_CEILING_S)


class StageDeadlineExceeded(TimeoutError):
    """Raised by `check_stage_deadline` when a stage outruns its wall-clock
    budget. Subclasses TimeoutError so the existing stage-isolation handlers
    (which already catch TimeoutError/Exception and convert it into a
    zero-score StageResult) need no change."""


def check_stage_deadline(world, where: str) -> None:
    """Cooperative wall-clock abort for a stage, checked at `where`.

    2026-08-14: this exists because NEITHER existing mechanism actually
    bounds a stage in the Isaac Sim Kit runtime.

    - `signal.alarm` (scripts/task3/run_task3_video_mainthread.py) fires but
      `omni.kit.async_engine` swallows the exception when it lands inside
      its own callback dispatch -- observed 2026-08-14, and recorded in
      plans/GOTCHAS.md.
    - `Thread.join(budget_s)` (orchestrator._run_stage_isolated) does return,
      but is followed by an unbounded `worker.join()` that waits for a thread
      which never stops.

    The consequence was not a slow episode but NO EPISODE AT ALL: a stage
    that never returns means `run_task3.run_one` never reaches its
    `result.as_json()` print, so the run produces no score rather than a low
    one. Three separate 2026-08-14 submission runs (45-112 min) were killed
    with zero output for exactly this reason.

    Cooperative checking is the mechanism that works here: no signals, no
    second thread, nothing for the Kit event loop to swallow. Granularity is
    one skill invocation / one ranked candidate, which is what the call
    sites choose. `world` without a deadline set (the unchanged default, and
    every MockWorld/CPU test) is a no-op.
    """
    deadline = getattr(world, "stage_deadline_monotonic", None)
    if deadline is None:
        return
    import time as _time

    remaining = deadline - _time.monotonic()
    if remaining > 0:
        return
    raise StageDeadlineExceeded(
        f"stage wall-clock budget exhausted at {where} "
        f"({-remaining:.0f}s past deadline)"
    )
