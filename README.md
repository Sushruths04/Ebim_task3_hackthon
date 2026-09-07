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

The default task is the cup onto the plate. To select another object and
destination:

```bash
docker run --rm --network host -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=... -e VISION_API_KEY=... \
  ebim-task3 --object "the rim of the bowl" --onto "the rim of the plate"
```

## Requirements

| | |
|---|---|
| Robot | dual 7-DoF arms on a shared vertical rail, parallel gripper, mobile base; ROS 2 Humble control stack running and accepting goals |
| Cameras | head and wrist cameras publishing |
| **Network** | **required at run time** — see below |
| Host | share the robot's ROS 2 network with `--network host` |
| **Safety** | **an operator must be holding the emergency stop.** The container asserts this on start. |

> **Network access is required.** The policy calls a vision service at
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
docker run --rm ebim-task3 --selftest    # loads the policy and exits; moves nothing
docker run --rm ebim-task3 --version     # build stamp
```

## Exit codes

| code | meaning | action |
|---|---|---|
| `0` | the task completed | — |
| `3` | the image is incomplete | contact us |
| `4` | vision service unreachable within 8 s | check `VISION_ENDPOINT` and egress |
| `5` | no robot found on the ROS network | check the control stack, `ROS_DOMAIN_ID`, and `--network host` |
| other | a stage failed | the last lines of output name the stage |

Every run prints its progress to stdout. If something fails, those lines are
the diagnosis — please include them if you contact us.

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to
publish the result. No other use is permitted.

## Contact

Include the output of `--version` and the last 20 lines of the run.
