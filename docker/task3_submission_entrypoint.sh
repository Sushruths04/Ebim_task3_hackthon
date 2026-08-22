#!/usr/bin/env bash
set -euo pipefail

# Entrypoint for the self-contained submission image (Dockerfile.submission).
#
# docker/task3_entrypoint.sh (the older dev-container entrypoint) calls
# scripts/task3/run_episode.py -- per handoff (see "Orchestrator already
# exists" finding), that is the WRONG file; it is not the working 4-stage
# orchestrator. The real one is task3_pipeline.run_task3, invoked here the
# same way the GPU verification runs on this project already do (via
# isaaclab.sh -p -m ...).

repo_root=/workspace/EBiM_Challenge
seed="${TASK3_SEED:-42}"
head_placement="${TASK3_HEAD_PLACEMENT:-a}"
out_dir="${TASK3_OUT_DIR:-$repo_root/outputs/task3_pipeline}"
memory="${TASK3_MEMORY:-$out_dir/param_memory.json}"

# 2026-08-14 update (supersedes the 2026-08-09 default below): that
# default predates the base-drive fix and the full Stage 1 pipeline that
# landed on main tonight, both proven on real GPU in
# plans/SUBMISSION_RUN_CHECKLIST_2026-08-14.md's run
# (outputs/submission_run_2026-08-14/). Default is now the real,
# unflagged pipeline -- Stage 1 (kitchen->dining) then Stage 4
# (dining->sink), real navigation and real grasp, all objects -- not the
# skip_navigation/skip_grasp/single-object shortcut this default used to
# take. Override via env vars for a narrower/faster attempt if needed.
#
# 2026-08-09 default, kept for provenance: Stage 4 only,
# stage4-objects=spoon2, navigation and grasp skipped
# (proofs/2026-08-09_o1_east_verify/spoon2_seed7_east.log, commit
# 8e3fdd9). Real GPU result, n=1: total_score 2/4. Honest reliability
# over 4 repeats: [2, 0, 0, 1] -- not proven reliable, just the
# best-evidenced choice available at the time.
order="${TASK3_ORDER:-1,4}"
skip_navigation="${TASK3_SKIP_NAVIGATION:-0}"
skip_grasp="${TASK3_SKIP_GRASP:-0}"
stage4_objects="${TASK3_STAGE4_OBJECTS:-}"
# REV16 Phase C: default OFF, unverified on GPU as of this commit -- see
# task3_autonomy/perception_grasp.py's module docstring. Any failure
# falls back to the existing constant grasp path the same run, so
# setting this to 1 before it is GPU-validated is safe, just unproven.
perception_grasp="${TASK3_PERCEPTION_GRASP:-0}"
# Live Gemini ER-2 grasp pose (position AND approach orientation) per
# attempt, DEFAULT ON. Without it the shipped container runs the old
# hardcoded straight-down wrist for every object, which is the thing this
# whole 2026-08-14/15 effort replaced -- an evaluator would grade a
# pipeline that has none of it. It degrades safely: no GEMINI_API_KEY, no
# network, a malformed answer or a grasp point too far from the object all
# fall back to the previous path the same attempt, logged via the
# `live_er_grasp` WORLD_ISAAC_DBG phase.
live_er_grasp="${TASK3_LIVE_ER_GRASP:-1}"
# Freeze the gripper's close target on contact instead of driving to fully
# shut. Also default ON -- see arms.run_gripper_close_ramp.
close_hold_on_contact="${TASK3_CLOSE_HOLD_ON_CONTACT:-1}"

if [[ "${1:-}" == "bash" || "${1:-}" == "shell" ]]; then
  exec /bin/bash
fi

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

args=(
  --seed "$seed"
  --head-placement "$head_placement"
  --memory "$memory"
  --out-dir "$out_dir"
)
[[ -n "$order" ]] && args+=(--order "$order")
[[ "$skip_navigation" == "1" ]] && args+=(--skip-navigation)
[[ "$skip_grasp" == "1" ]] && args+=(--skip-grasp)
[[ -n "$stage4_objects" ]] && args+=(--stage4-objects "$stage4_objects")
[[ "$perception_grasp" == "1" ]] && args+=(--perception-grasp)
[[ "$live_er_grasp" == "1" ]] && args+=(--live-er-grasp)
[[ "$close_hold_on_contact" == "1" ]] && args+=(--close-hold-on-contact)

exec /workspace/isaaclab/isaaclab.sh -p -m task3_pipeline.run_task3 "${args[@]}"
