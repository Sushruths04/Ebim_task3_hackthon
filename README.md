# EBiM Task 3 — Assisted Living & Feeding

Autonomous table setting on a dual-arm mobile manipulator.

## Build and run

```bash
docker build -t ebim-task3 .

docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=http://<host>:8643 \
  -e VISION_API_KEY=<supplied separately> \
  ebim-task3
```

To select a different object and destination:

```bash
docker run --rm --network host -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=... -e VISION_API_KEY=... \
  ebim-task3 --object "the rim of the bowl" --onto "the rim of the plate"
```

## What it does

From one command and with no operator, the robot parks its idle arm, homes the
working arm, drives until the pick can run, locates the object and its
destination, picks the object at its rim, carries it, places it on the
destination's measured surface, and returns to a known configuration. Each
stage reports its measurement and the run stops at the first stage that fails.

The grasp is a rim-wall grasp: one jaw inside the rim and one outside, closing
across the wall. It is the method that generalises, because a bowl and a plate
are both wider than the gripper's stroke.

## Requirements

| | |
|---|---|
| Robot | dual 7-DoF arms on a shared vertical rail, parallel gripper, mobile base; ROS 2 Humble control stack running and accepting goals |
| Cameras | one head camera and one wrist depth camera, publishing |
| **Network** | **required at run time.** The policy calls a vision service over HTTP; see below. |
| Host | share the robot's ROS 2 network with `--network host` |
| **Safety** | **an operator must be holding the emergency stop.** The container asserts this on start. |

> **Network access is required.** Perception is a call to a vision service at
> `VISION_ENDPOINT`. With no route to it the container exits 4 immediately and
> does not move the robot. It will not run under `--network none`.

## Environment

| variable | default | meaning |
|---|---|---|
| `ROS_DOMAIN_ID` | `0` | must match the robot's control stack |
| `VISION_ENDPOINT` | — | vision service URL; **required** |
| `VISION_API_KEY` | — | supplied separately; not baked into the image |
| `TMR_WS` | unset | overlay ROS workspace, if the robot needs one |
| `EBIM_REQUIRE_VISION` | `1` | set `0` to start even if the vision service does not answer |

## Check the image before a run

```bash
docker run --rm ebim-task3 --selftest    # loads its own modules and exits
docker run --rm ebim-task3 --version     # build stamp
```

## Exit codes

| code | meaning | action |
|---|---|---|
| `0` | the task completed | — |
| `3` | calibration or map missing from the image | the image is broken; contact us |
| `4` | vision service unreachable within 8 s | check `VISION_ENDPOINT` and egress |
| `5` | no robot found on the ROS network | check the control stack, `ROS_DOMAIN_ID`, and `--network host` |
| other | a stage failed | the last lines name the stage and its measurement |

Every run prints its stages and measurements to stdout. If something fails,
those lines are the diagnosis — please include them if you contact us.

## Behaviour that is correct, but may look wrong

- **It reports that it cannot measure whether the object rose after the grasp.**
  That is what success looks like here: a held object sits inside the wrist
  camera's near blind zone, so the hold is confirmed from the head camera.
- **The idle arm is parked before anything else happens.** Both arms share one
  vertical rail, so the idle arm must be moved clear before any descent.
- **The robot returns through its home configuration between picking and
  placing.** Deliberate — it is what keeps the arm out of a self-collision.

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to
publish the result. No other use is permitted.

## Contact

Include the output of `--version` and the last 20 lines of the run.
