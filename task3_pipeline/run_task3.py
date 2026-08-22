# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Entry point for one (or a matrix of) autonomous Task 3 episode(s).

    # CPU smoke test / logic demo -- no Isaac needed:
    python -m task3_pipeline.run_task3 --mock --seed 42 --head-placement a

    # Full unattended matrix (the "18 manual runs", automated):
    python -m task3_pipeline.run_task3 --mock --matrix

    # Real robot (on an RTX host inside the Isaac container):
    python -m task3_pipeline.run_task3 --seed 42 --head-placement a
        --record-video

The only difference between mock and real is which WorldAdapter is constructed;
the orchestrator, verifier, memory and retry logic are identical.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import statistics
import sys
import traceback
from typing import Any

from task3_pipeline import config
from task3_pipeline.orchestrator import Task3Pipeline
from task3_pipeline.world import MockWorld

HEAD_PLACEMENTS = tuple("abcdefghi")


def _make_world(args, simulation_app: Any = None):
    if args.mock:
        return MockWorld(seed=args.seed, head_placement=args.head_placement)
    # Real robot: imported lazily so CPU/mock runs never touch Isaac.
    from task3_pipeline.world_isaac import IsaacWorld  # noqa: WPS433

    return IsaacWorld(
        simulation_app=simulation_app,
        record_video=args.record_video,
        out_dir=args.out_dir,
        skip_navigation=args.skip_navigation,
        skip_grasp=args.skip_grasp,
        use_curobo_stance=not args.no_curobo_stance,
        stage4_objects=_parse_stage4_objects(args.stage4_objects),
        stage1_objects=_parse_stage1_objects(args.stage1_objects),
        close_hold_on_contact=args.close_hold_on_contact,
        select_nearer_arm_side=args.select_nearer_arm_side,
        push_perception_targets=args.push_perception_targets,
        curobo_rate_both_arms=args.curobo_rate_both_arms,
        reach_gate_enabled=not args.no_reach_gate,
        push_stance_navigate_budget_s=args.push_stance_navigate_budget_s,
        use_ranked_grasp=not args.no_ranked_grasp,
        perception_grasp=args.perception_grasp,
        live_er_grasp=args.live_er_grasp,
        curobo_grasp=args.curobo_grasp,
        gripper=args.gripper,
    )


def _parse_object_subset(raw: str | None, flag: str) -> tuple[str, ...] | None:
    """Parse a comma-separated object subset for `flag`. Pure/CPU-testable
    (no Isaac import).

    Returns None for "all objects" (the unchanged default) so callers can
    pass the result straight through.
    """
    if not raw:
        return None
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    unknown = [n for n in names if n not in config.STAGE1_OBJECTS]
    if unknown:
        raise SystemExit(
            f"{flag}: unknown object(s) {unknown}; "
            f"valid names are {list(config.STAGE1_OBJECTS)}"
        )
    return names or None


def _parse_stage4_objects(raw: str | None) -> tuple[str, ...] | None:
    """Parse --stage4-objects. Thin wrapper over `_parse_object_subset`,
    kept as its own name because the tests import it directly."""
    return _parse_object_subset(raw, "--stage4-objects")


def _parse_stage1_objects(raw: str | None) -> tuple[str, ...] | None:
    """Parse --stage1-objects. Stage 1's twin of the above (2026-08-14)."""
    return _parse_object_subset(raw, "--stage1-objects")


def _app_launcher_config(args) -> dict[str, Any]:
    """The AppLauncher kwargs for a real (non-mock) run -- pure/CPU-testable
    so W0.1's gate (enable_cameras follows --record-video, now also
    --perception-grasp -- REV16 Phase C.4: the segmentation/depth
    annotators need rendering active, and it must be enabled here, at
    AppLauncher time on the main thread, not via a mid-episode
    enable_extension() call from the stage worker thread -- that hung for
    10+ minutes on omni.kit.material.library startup, GPU-verified
    2026-08-08) doesn't need isaaclab installed to verify. Cameras only
    when actually needed:
    enable_cameras makes sim.step() run full app updates whose USD sync can
    interfere with tensor-API joint targets (2026-07-17 investigation,
    verify_navigate.py:78-80) -- and it triples wall time (handoff.md sec
    18.2). CORRECTION (sec 18.2): this used to be hard-coded True on every
    run, paying that 2-3x tick-cost penalty even when nothing was recorded.
    """
    return {
        "headless": True,
        "enable_cameras": bool(
            args.record_video
            or args.perception_grasp
            or args.live_er_grasp
        ),
        "livestream": -1,
    }


def _make_pipeline(world, args) -> Task3Pipeline:
    kwargs: dict[str, Any] = {}
    if args.stage_timeout_s is not None:
        kwargs["stage_wallclock_budget_s"] = args.stage_timeout_s
    if args.measured_s_per_tick is not None:
        kwargs["measured_s_per_tick"] = args.measured_s_per_tick
    return Task3Pipeline(world, memory_path=args.memory, **kwargs)


def run_one(args, simulation_app: Any = None) -> None:
    world = _make_world(args, simulation_app)
    pipe = _make_pipeline(world, args)
    kwargs = {}
    if args.order:
        kwargs["order"] = tuple(int(s) for s in args.order.split(","))
    result = pipe.run_episode(
        seed=args.seed, head_placement=args.head_placement, **kwargs
    )
    print(result.as_json(), flush=True)


def run_matrix(args, simulation_app: Any = None) -> None:
    world = _make_world(args, simulation_app)
    pipe = _make_pipeline(world, args)
    scores, pcts = [], []
    for hp in HEAD_PLACEMENTS:
        for seed in range(args.seeds):
            r = pipe.run_episode(seed=seed, head_placement=hp)
            scores.append(r.total)
            pcts.append(r.pct)
            print(r.as_json(), flush=True)
    n = len(pcts)
    passed = sum(1 for p in pcts if p >= 0.70)
    print(
        "MATRIX_SUMMARY "
        + str(
            {
                "runs": n,
                "median_pct": round(statistics.median(pcts), 3),
                "mean_pct": round(statistics.mean(pcts), 3),
                "fraction_ge_70pct": round(passed / n, 3),
            }
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous Task 3 pipeline runner"
    )
    p.add_argument(
        "--mock", action="store_true", help="use MockWorld (no Isaac)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--head-placement", choices=HEAD_PLACEMENTS, default="a")
    p.add_argument("--matrix", action="store_true", help="run 9 x N matrix")
    p.add_argument(
        "--seeds", type=int, default=10, help="seeds per placement in --matrix"
    )
    p.add_argument(
        "--memory", default="outputs/task3_pipeline/param_memory.json"
    )
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--out-dir", default="outputs/task3_pipeline")
    p.add_argument(
        "--skip-navigation",
        action="store_true",
        help=(
            "spawn the robot at the kitchen-local rotation-safe stance "
            "world_isaac.py's reset() uses for verify_grasp_lift-style "
            "fast arm iteration, instead of the full dining-room spawn -- "
            "exercises Stage 3/4 (both kitchen-local) with a short final "
            "navigate_to() leg instead of the full doorway route. Ignored "
            "with --mock. Result JSON records skip_navigation: true/false."
        ),
    )
    p.add_argument(
        "--skip-grasp",
        action="store_true",
        help=(
            "C0 diagnostic (handoff sec 49 / plan CUROBO_PIVOT_PLAN sec 5 "
            "R2.2): stage 4's plan_stage4 skips its grasp() call entirely "
            "and goes straight from reach() to the contact push, isolating "
            "whether the failing grasp is what flings objects off the "
            "counter before the push ever runs. Ignored with --mock."
        ),
    )
    p.add_argument(
        "--no-curobo-stance",
        action="store_true",
        help=(
            "GATE B0 (handoff sec 68): disable the cuRobo batch-IK stance "
            "search and use the old fixed-radius navigation.stance_for() "
            "directly -- for A/B comparison only. Default is the cuRobo "
            "search ON, since sec 64-67 proved the fixed radius is the "
            "root cause of most Stage 1/4 reach failures. Ignored with "
            "--mock."
        ),
    )
    p.add_argument(
        "--no-ranked-grasp",
        action="store_true",
        help=(
            "REV12 T6: disable plan_stage1's ranked-candidate grasp path "
            "(world.reach_and_grasp_ranked, side chosen by "
            "world._select_arm_side) and fall back to the old "
            "hardcoded-'right'-arm reach()+grasp() pair -- for A/B "
            "comparison (REV12 T8) only. Default is ranked-grasp ON. An "
            "object with no ranked candidates file behaves identically "
            "either way -- reach_and_grasp_ranked's own fallback is "
            "byte-identical to calling reach()/grasp() directly. Ignored "
            "with --mock (MockWorld has neither method)."
        ),
    )
    p.add_argument(
        "--perception-grasp",
        action="store_true",
        help=(
            "REV16 Phase C: try a perception-derived grasp point "
            "(instance-segmentation mask centroid + principal-axis grasp "
            "+ minor-axis width check, task3_autonomy/perception_grasp.py) "
            "in reach() before falling back to the existing hand-fitted "
            "cup/object constant offsets. Default OFF -- unchanged "
            "behavior; any failure (annotators not ready, empty mask, no "
            "IK-feasible side, or any exception) falls back to the "
            "constant path the same run, logged via the "
            "'perception_grasp_target' WORLD_ISAAC_DBG phase. Ignored "
            "with --mock."
        ),
    )
    p.add_argument(
        "--live-er-grasp",
        action="store_true",
        help=(
            "Ask Gemini Robotics ER-2, LIVE and once per grasp attempt, for "
            "this object's grasp POSITION and APPROACH ORIENTATION, instead "
            "of commanding a fixed straight-down wrist for every object. "
            "Measured against the organisers' own recorded demonstrations "
            "(FK'd through our URDF, plans/PROGRESS.md), a straight-down "
            "command is 52-84 degrees away from the approach that actually "
            "works on this robot, which is why the reach residual is flat "
            "across every spine height instead of shrinking. Needs "
            "GEMINI_API_KEY_PRIMARY (and optionally _SECONDARY). Default "
            "OFF. Any failure -- no key, no network, a malformed answer, a "
            "bad depth reading, or a grasp point too far from the object -- "
            "falls back to the existing path the same attempt, logged via "
            "the 'live_er_grasp' WORLD_ISAAC_DBG phase. Implies the camera "
            "extensions, same as --perception-grasp. Ignored with --mock."
        ),
    )
    p.add_argument(
        "--curobo-grasp",
        action="store_true",
        help=(
            "Plan the approach and grasp with cuMotion "
            "(curobo.motion_planner.MotionPlanner.plan_grasp) and fly the "
            "returned joint trajectories, instead of servoing at a Cartesian "
            "target leg by leg. Motivation, measured: ER-2 puts the grasp "
            "point 0.048 m from the object and the pads finish 0.132-0.165 m "
            "away, because the servo ends 0.05-0.07 m from its own target "
            "with IK solving on every tick -- the arm misses a target it "
            "already knows. A planned trajectory ends AT the solution. "
            "Default OFF; any planning or execution failure falls back to "
            "the servo path the same attempt, logged via the "
            "'curobo_grasp_plan' / 'curobo_grasp_flown' phases. NOTE: the "
            "robot YAML currently has no collision spheres, so this plans "
            "kinematically, not around obstacles."
        ),
    )
    p.add_argument(
        "--select-nearer-arm-side",
        action="store_true",
        help=(
            "P5 (plans/LOOP_PROMPT_VM_A.md rev 2): instead of hardcoding "
            "the right arm for every Stage 4 push/carry, pick whichever "
            "arm's live target_norm_from_arm_base_m is smaller for the "
            "object's current position. Default OFF -- unchanged 'right' "
            "behavior, trivially revertible. Ignored with --mock."
        ),
    )
    p.add_argument(
        "--push-perception-targets",
        action="store_true",
        help=(
            "Q3 (SYNC 22-24): use task3_autonomy/perception_targets.py's "
            "ER/BBOX-ranked, IK-screened contact height for bowl2/spoon2's "
            "push ONLY -- the two objects N2 proved it wins for offline. "
            "NOT cup (N3 refuted it) or plate2 (hardcoded won). One "
            "batched ER call per episode, cached, never aborts the stage "
            "on failure (falls back to the existing geometry). Default "
            "OFF. Ignored with --mock."
        ),
    )
    p.add_argument(
        "--curobo-rate-both-arms",
        action="store_true",
        help=(
            "Q5 (SYNC 21/25): CuroboStanceSearch used to rate every "
            "candidate stance against the right arm's base only "
            "(curobo_stance.py, originally :313) -- flagged independently "
            "by VM B twice and this session's own P5 circularity note. "
            "Rates against both arm bases and keeps a candidate reachable "
            "by EITHER. Default OFF -- unchanged right-only behavior. "
            "Ignored with --mock or --no-curobo-stance."
        ),
    )
    p.add_argument(
        "--no-reach-gate",
        action="store_true",
        help=(
            "T5 (LOOP_PROMPT_VM_A_REV4.md): disable the Q2/Q4 reach-limit "
            "pre-flight gate (_reach_limit_exceeded) for an A/B test. "
            "Default OFF (gate stays ON, unchanged behavior) -- the gate "
            "is correct engineering that may also be refusing the exact "
            "out-of-reach attempts that produced the project's only point "
            "(handoff sec 105). Ignored with --mock."
        ),
    )
    p.add_argument(
        "--push-stance-navigate-budget-s",
        type=float,
        default=25.0,
        help=(
            "Correction to the T5a REFUTED verdict (plans/SYNC.md "
            "2026-08-04 ~14:45 UTC): navigate_to has never arrived at a "
            "curobo_stance_for candidate in any run recorded since T4 -- "
            "T4's fix picks candidates ~2-3m away and this budget has "
            "always been hardcoded at 25.0s. A/B this without a code "
            "change. Default 25.0 (unchanged behavior). Ignored with "
            "--mock."
        ),
    )
    p.add_argument(
        "--order",
        default=None,
        help=(
            "comma-separated stage numbers to run, e.g. '2' or '3' -- lets "
            "a single stage be isolated (skipping Stage 1's slow per-object "
            "loop) without a full reset()/scene rebuild of its own. "
            "Ignored with --matrix. Default: full 1,2,3,4 chain."
        ),
    )
    p.add_argument(
        "--stage-timeout-s",
        type=float,
        default=None,
        help=(
            "override the per-stage wall-clock budget with ONE flat number "
            "for every stage, bypassing config.STAGE_TICK_BUDGETS entirely "
            "-- for single-stage isolating experiments that want one known "
            "wall-clock ceiling (e.g. handoff sec 4.65's `--order 1 "
            "--stage-timeout-s 1200`), not for normal runs. Normal runs "
            "should set --measured-s-per-tick instead so each stage's "
            "ceiling is derived from its own tick budget."
        ),
    )
    p.add_argument(
        "--stage4-objects",
        default=None,
        help=(
            "REVIEW #9 (handoff sec 76): comma-separated subset of "
            "config.STAGE1_OBJECTS that stage 4 will ATTEMPT, e.g. "
            "'cup'. The scorer is unchanged -- score_stage(4) still scores "
            "all 4 objects, so max_score stays 4 -- this only bounds the "
            "work. Needed because stage 4's tick budget (460,800) is 4.0x "
            "more than HARD_JOIN_CEILING_S=3600 s can execute at the "
            "measured 0.0316 s/tick, so the full 4-object loop is always "
            "killed by TimeoutError before its retry grid ever runs. Since "
            "grading.py scores 1 point PER object, one object in the sink "
            "is enough to close GATE B1. Default: all 4 (unchanged)."
        ),
    )
    p.add_argument(
        "--close-hold-on-contact",
        action="store_true",
        help=(
            "2026-08-14: freeze the gripper's commanded close target the "
            "moment contact is detected, instead of continuing to drive "
            "toward fully-closed. REV13 T4-followup built this and left it "
            "an explicit opt-in 'pending live GPU verification'; this flag "
            "is what makes that verification runnable. Motivation, measured "
            "the same day: the re-centered wrist sits 0.0149 m from the "
            "grasp point, contact fires at tick 17 of 300, and by the hold "
            "check the object is 0.107 m away -- the gripper spends the "
            "remaining 283 ticks pushing the object out of its own grasp. "
            "Gated by contact_freeze_max_target_rad (0.65) so it cannot "
            "re-trigger T4's false freeze on servo lag at the start of the "
            "ramp. Default OFF (unchanged)."
        ),
    )
    p.add_argument(
        "--stage1-objects",
        default=None,
        help=(
            "2026-08-14: stage 1's twin of --stage4-objects, added for the "
            "same structural reason. Comma-separated subset of "
            "config.STAGE1_OBJECTS that stage 1 will ATTEMPT, e.g. "
            "'plate2'. The scorer is unchanged -- score_stage(1) still "
            "scores all 4 objects and max_score stays 4 -- this only bounds "
            "the work. Needed because stage 1 loops all 4 objects with a "
            "full retry budget each, so one unreachable object makes the "
            "stage unable to RETURN, and run_one() then never reaches its "
            "result.as_json() print: the episode yields no score at all "
            "rather than a low one. Default: all 4 (unchanged)."
        ),
    )
    p.add_argument(
        "--measured-s-per-tick",
        type=float,
        default=None,
        help=(
            "measured wall-seconds-per-tick used to derive each stage's "
            "wall-clock ceiling from config.STAGE_TICK_BUDGETS (handoff "
            "sec 18.1c/18.6 W0.5) -- e.g. worker N1's measured number. "
            "Defaults to config.MEASURED_S_PER_TICK_FALLBACK (an upper "
            "bound from a proof bundle whose camera setting is "
            "undocumented) until a session measures a real one."
        ),
    )
    p.add_argument(
        "--gripper",
        default=None,
        choices=(None, "robotiq", "panda"),
        help=(
            "REV20 §3: which robot USD to load. Default (None) is the "
            "exact file this repo has always actually loaded "
            "(assets/mobile_fr3_duo_v0_2.usd) -- NOT the same as "
            "gripper_profiles.py's own 'panda' entry (the organizers' thin "
            "reference layer, untested here). 'robotiq' loads the real "
            "competition robot (Robotiq 2F-85, D405 wrist cams); every "
            "grasp constant in this repo was calibrated against the "
            "default's fingers, so this is opt-in until measured parity."
        ),
    )
    return p


def main(argv=None) -> None:
    # sec 19b W1.2: all-thread Python stack dumps with no CAP_SYS_PTRACE and
    # no container change (§19.3's blocker was the only way to get a frame
    # before this). dump_traceback_later fires into stderr (captured by the
    # run's redirected log) every 10 min for the life of the process; SIGUSR1
    # additionally allows an on-demand dump via `docker exec ... kill -USR1
    # <pid>` without waiting for the timer.
    faulthandler.enable()
    faulthandler.dump_traceback_later(600, repeat=True, exit=False)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    args = build_parser().parse_args(argv)
    if args.mock:
        if args.matrix:
            run_matrix(args)
        else:
            run_one(args)
        return

    # Real robot: IsaacWorld requires an AppLauncher app object to exist
    # BEFORE it is constructed (world_isaac.py's own module docstring).
    # [OBSERVED, Phase-1 Worker A pre-flight, handoff sec 4.55] this was
    # never wired here at all -- `run_task3.py --seed ... --record-video`
    # (no --mock) crashed immediately with "IsaacWorld requires
    # simulation_app" before running a single tick. Mirrors the AppLauncher
    # config + shutdown pattern already used by run_stage4_cleanup.py /
    # run_stage2_feeding.py.
    if not args.no_curobo_stance:
        # BLOCKER 3 (handoff sec 51/58): warp must be imported before
        # AppLauncher constructs the Isaac Sim app, or cuRobo's later
        # import hangs. CuroboStanceSearch (task3_autonomy/curobo_stance.py)
        # is built lazily inside IsaacWorld._stance_for, well after
        # AppLauncher -- so warp has to be front-loaded here instead.
        sys.path.insert(0, "/workspace/curobo_spike")
        import warp as wp  # noqa: F401

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(_app_launcher_config(args))
    simulation_app = app_launcher.app
    try:
        if args.matrix:
            run_matrix(args, simulation_app)
        else:
            run_one(args, simulation_app)
    except BaseException:
        # GATE B7 (handoff sec 63/71): simulation_app.close() itself hangs
        # (a documented Kit shutdown issue, unrelated to this exception) --
        # calling it here used to mask a real exception for hours behind
        # what looked like an ordinary slow shutdown. Print the traceback,
        # flush, then hard-exit WITHOUT calling close() or re-raising --
        # os._exit() bypasses Python's normal unwind (atexit/finally), so
        # nothing downstream ever depended on this function actually
        # raising; the traceback above is the only signal that ever
        # mattered, and it is now guaranteed to reach the log before the
        # process disappears.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        # Avoid Kit's documented shutdown hang (same workaround as the
        # other Isaac scripts) -- results are already printed above.
        os._exit(0)


if __name__ == "__main__":
    main()
