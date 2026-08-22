# EBiM Benchmark — Task 3 Submission

Autonomous tabletop manipulation for the EBiM Autonomous Robot Benchmark
(Rulebook 1.0), Task 3. Mobile dual-arm Franka FR3 with Robotiq 2F-85
grippers, in Isaac Sim.

The full write-up, including measured results and current limitations, is in
[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) (PDF alongside it).

## What this is

A physics-only manipulation pipeline: no imitation learning, no VLA, no
trained policy in the control loop. The robot perceives an object with its own
head camera, derives a grasp from measured geometry, plans a collision-aware
trajectory with cuRobo, and closes on verified physical contact.

## Build

```bash
docker build -t ebim-task3:submission .
```

The base image (`nvcr.io/nvidia/isaac-lab`) is pinned **by digest**, so a
future repush of the tag cannot change what gets built. It is anonymously
pullable — no `docker login` required.

## Run

Chained Stage 1 → Stage 4 (table setup, then cleanup to the sink), which is
the flow this submission targets:

```bash
docker run --rm --gpus all \
  -v "$PWD/outputs:/workspace/EBiM_Challenge/outputs" \
  ebim-task3:submission \
  --order 1,4 --seed 42 --head-placement a
```

All four stages:

```bash
docker run --rm --gpus all \
  -v "$PWD/outputs:/workspace/EBiM_Challenge/outputs" \
  ebim-task3:submission \
  --seed 42 --head-placement a
```

Useful flags: `--order` (which stages, comma-separated), `--seed`,
`--head-placement {a,b,c}`, `--record-video`, `--out-dir`.

An episode prints a JSON `EPISODE_RESULT` line with per-stage scores and
writes it under `--out-dir` (default `outputs/task3_pipeline`).

## Requirements

- NVIDIA GPU with the container toolkit (`--gpus all`). Developed and measured
  on a single L4 (23 GB).
- ~60 GB free disk for the image.

## Tests (CPU, no GPU needed)

The perception, grasp-geometry and state-machine logic are unit-testable
without Isaac:

```bash
python -m pytest task3_pipeline/tests task3_autonomy/tests -q
```

## Layout

| Path | What |
|---|---|
| `task3_pipeline/` | Orchestrator, stages, scoring, world adapter, perception |
| `task3_autonomy/` | Navigation, arm control, grasp planning, GraspGenX client |
| `scripts/scenes/` | Scene construction and robot configuration |
| `docker/` | Container entrypoint |
| `TECHNICAL_REPORT.md` | Method, results, limitations |

## Stage coverage

| Stage | What it is | State |
|---|---|---|
| 1 | Table setup — objects to assigned seats | Implemented; validated (`cup_lift_m` 0.1112, 3.0 s hold) |
| 2 | Feeding — scoop and hold | Implemented; not validated |
| 3 | Bean recovery — bowl-tilt pour | Implemented; not validated |
| 4 | Cleanup — objects to the sink | Implemented |

Stages 2 and 3 are implemented but unvalidated, which is why the documented
run above uses `--order 1,4`. See the report for the measured evidence behind
each of these.
