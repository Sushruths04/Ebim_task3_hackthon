# Task 3 — Autonomous Self-Correcting Pipeline

An end-to-end, fully autonomous controller for all four Task 3 stages, with an
**integrated verify → diagnose → adjust → retry → remember loop** that replaces
the manual "watch a GIF, edit one constant, rerun" workflow. No human at run
time, no training required to hit the target.

It is a thin orchestration layer on top of the proven primitives in
`task3_autonomy/` (navigation, dual-arm control, the fail-closed chain FSM) —
it reuses them, it does not rewrite them.

**Status:** the brain (orchestration, verifier, memory, retry, stage plans) is
implemented and unit-tested on CPU. A 90-run mock matrix that starts from
deliberately-wrong defaults self-corrects to **median 93.8%, 100% of runs
≥ 70%**. The only remaining work is wiring `world_isaac.py` to the real sim.

```
python -m pytest task3_pipeline/tests -q          # or:
PYTHONPATH=. python -B task3_pipeline/tests/test_pipeline.py
PYTHONPATH=. python -m task3_pipeline.run_task3 --mock --matrix   # 90-run demo
```

---

## Why the old pipeline stalled

The scripted FSM is the *correct* competition point-scorer (autonomous ≠
learned). What was broken was the **loop around it**: a human was the verifier —
~18 manual GPU runs to get one grasp close, with every hard-won fact trapped as
prose in `AGENT_STATE.md`. Hand-tuned constants didn't generalise across seeds
or the 9 head placements, and the Stage-4 "grasp" failure was really a
*reachability* failure (the right arm can't reach the cup Y-offset from the
east stance → weak cage → IK fling on lift).

This package fixes the loop, not just the constant.

## The idea in one picture

```
              ┌──────────────── ParamMemory (JSON) ────────────────┐
              │  per (skill, head_placement, object):               │
              │  best-known params · failed params · success stats  │
              └───────▲───────────────────────────────┬────────────┘
     records outcome  │                      queried   │ best-first
                      │                                 ▼
   ┌───────────┐   ┌──┴──────────┐   ┌──────────────────────────┐
   │  Stage    │──▶│ SelfCorrect │──▶│ RetryPolicy              │
   │  plan     │   │ .run(skill) │   │ next params from grid,   │
   │ (scorer-  │   │             │◀──│ diagnosis-prioritised    │
   │  aligned) │   └──────┬──────┘   │ (IK_FAIL → flip stance…) │
   └───────────┘          │ invoke   └──────────────────────────┘
                          ▼
                  ┌───────────────┐  metrics   ┌──────────────┐
                  │ WorldAdapter  │──────────▶ │ auto-verifier│
                  │ Mock / Isaac  │            │ classify()   │
                  └───────────────┘            └──────────────┘
        Task3ChainFSM sequences stages 1→2→3→4, fail-closed on safety.
```

The self-correction loop *is* Stages A + B of the `FUTURISTIC_PIPELINE`
research design (agentic verify + failure memory) — built here because it also
wins the competition. Slow-loop learning (IL / residual RL) plugs in later at
the same `WorldAdapter` boundary without touching this code.

## Files

| File | Role |
|---|---|
| `config.py` | Scorer-aligned geometry (mirrors `grading.py`), verifier thresholds, bounded parameter grids. **The only place to edit targets.** |
| `outcomes.py` | The **auto-verifier**: `classify(skill, metrics) → (SkillOutcome, diagnosis)`. Pure. |
| `memory.py` | Persistent **ParamMemory** (JSON). `best_params` / `failed_params` / `summary`. |
| `policy.py` | **RetryPolicy**: next params, best-first, diagnosis-prioritised, avoids known-fail combos. Pure. |
| `skills.py` | **SelfCorrectingSkill.run** — the fast loop around one primitive. |
| `stages.py` | The four **scorer-aligned stage plans** (strategy lives here). |
| `orchestrator.py` | **Task3Pipeline** — chains stages via `Task3ChainFSM`, emits `EPISODE_RESULT` JSON. |
| `world.py` | `WorldAdapter` Protocol + **MockWorld** (CPU, parameter-sensitive, encodes real Task-3 truths). |
| `world_isaac.py` | **IsaacWorld integration stub** — the one file that imports Isaac; each method's wiring is documented. |
| `seats.py` | Seat-target interface for Stage 1: `assigned_seats()` / `object_to_seat()`. Real coords are a LATER task (T1); stubbed with a documented mock today. |
| `run_task3.py` | CLI: single episode or 9×N matrix; `--mock` or real. |
| `tests/` | CPU unit tests (verifier, memory, policy, skill recovery, full episode, matrix). |

## Per-stage strategy (aligned to the organizer prose rules — see
`docs/TASK3_MASTER_EXECUTION_PLAN_2026-07-24.md`; `grading.py` is a dev
smoke-test only, not the objective truth)

- **Stage 1 — Table setup:** **NO tray.** Per object (plate, cup, bowl+beans,
  spoon — `config.STAGE1_OBJECTS`): navigate to a reach-safe kitchen stance,
  reach, grasp, carry/place at its assigned seat (`seats.py` resolves 3 of 6
  seats per episode; real coords are a LATER task, T1).
- **Stage 2 — Feeding:** scoop at a 30–45° entry with a deep drag, present
  **slowly** to the feed zone, hold ≥ 3 s. (Scorer: smooth + 3 s hold + beans
  on spoon. The head-force gate is a *hard fail* — approach slow, stop short.)
- **Stage 3 — Bean recovery:** pour the beans back into the recovery sphere
  **low and slow** — it's a ratio game (≥ 0.8 → 2 pts, ≥ 0.9 → 3, = 1.0 → 4).
- **Stage 4 — Cleanup:** honest grasp + place of each of the 4 utensils into
  the marked sink region (navigate → grasp → carry/place → verify). A
  controlled carry/place is legal and sufficient — the scorer checks the
  object ends in-region at height, not a held cage — but it follows a real,
  verified grasp (see `outcomes.classify_grasp`), not a physics exploit.

## Wiring it to Isaac (the only GPU-side work)

Implement each method in `world_isaac.py` to return the same metric keys as
`MockWorld`. The mapping is written out inline; in short:

1. `reset` → build the task3 scene (reuse `scene_robot_room_keyboard.py`).
2. `navigate_to` → loop `NavigateTo.compute` → `TmrBaseAdapter.apply_twist`.
3. `reach` → set stance, then `DualArmController.reach` (world→base frame).
4. `grasp` / `lift` → `DualArmController.grasp/lift` with the **proven
   `verify_grasp_lift.py` constants** (10/10) — and use its fling-free lift.
5. `carry_object_to` → base-carry / slide for Stage 4.
6. `scoop` / `feed_hold` / `pour` → spoon + bowl motions.
7. `score_stage` → call the official `grading.py` scorers with live poses.

Develop and debug the brain on CPU against `MockWorld`; spend GPU time only
making these methods return **real** measurements. Then:

```
PYTHONPATH=. python -m task3_pipeline.run_task3 --matrix --seeds 10   # unattended
```

The matrix populates `param_memory.json` with the best config per situation;
freeze on the first config whose `fraction_ge_70pct` clears your bar.

## How to improve it (the slow loop, later)

Everything above needs **zero training**. When the fast loop hits its ceiling,
the same `WorldAdapter` boundary is where learning bolts on, exactly as in the
`FUTURISTIC_PIPELINE` design:

1. **Log** every episode (obs, action, outcome) in LeRobot format — the
   `SkillReport`s already carry it.
2. **Imitation-learn** a small ACT / Diffusion policy for the weakest skill
   (grasp) from the logged successes; expose it as an alternative `grasp`
   method behind the same interface.
3. **Residual RL** (HiL-ResRL-style) on top of that policy in Isaac Lab.
4. Redeploy → collect more → repeat (DexFlyWheel). The orchestrator, verifier,
   memory and retry code do not change.

## Cost discipline

Write and unit-test all logic on CPU (this whole package runs without a GPU).
Isaac needs RT cores → develop on **L4 / L40S**, not A100. Log spend in
`docs/gpu_budget_log.md`.
