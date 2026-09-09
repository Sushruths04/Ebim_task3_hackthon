# EBiM Task 3 — Assisted Living & Feeding

Autonomous pick-and-place on a dual-arm mobile manipulator. One command, no
operator in the loop.

> **On an arm64 host (the Jetson companion), use the multi-architecture image
> instead** — this build is amd64 only. It carries the same pick-and-place plus
> the navigation: `ghcr.io/sushruths04/ebim-task3-mission:1.2.0`, see
> [README_MISSION.md](README_MISSION.md).
>
> **There is also a navigation build.** This README covers the submitted
> pick-and-place: the robot squares up to an object in front of it, picks it and
> places it. A second image adds driving between rooms — see
> [README_MISSION.md](README_MISSION.md), which opens with a test-run sequence.
> The build described here is the submission.

> **Running this on the testbed?** [OPERATING.md](OPERATING.md) has every
> command, both scenarios, and what the robot does at each step.

## Run

```bash
docker build -t ebim-task3 .

docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=<supplied to the organizers directly> \
  ebim-task3
```

The default task is the cup onto the plate. Any object and destination can be
named:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=<supplied to the organizers directly> \
  ebim-task3 --object "the rim of the bowl" --onto "the rim of the plate"
```

## Environment

| variable | required | default | meaning |
|---|---|---|---|
| `VISION_ENDPOINT` | **yes** | — | the vision service. **Sent to the organizers directly, not published here.** The container exits `4` before moving if this is unset or unreachable. |
| `VISION_API_KEY` | no | — | not needed when `VISION_ENDPOINT` is set; the service holds its own credential |
| `ROS_DOMAIN_ID` | no | `0` | must match the robot's control stack |
| `EBIM_REQUIRE_VISION` | no | `1` | set `0` to start even if the vision service does not answer |
| `TMR_WS` | no | unset | overlay ROS workspace, if the robot needs one |

The vision service is hosted and operated by us and is live. **There is no
credential for you to set** — the service holds its own. All the container needs
is the address and outbound HTTPS to reach it.

**The address is provided to the organizers directly rather than published in
this repository**, so that the service stays available for evaluation. Once you
have it, confirm it answers before a run:

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$VISION_ENDPOINT/health"   # expect 200
```

## Requirements

| | |
|---|---|
| Robot | dual 7-DoF arms on a shared vertical rail, parallel gripper, mobile base |
| Stack | ROS 2 Humble control stack running and accepting goals; cameras publishing |
| Network | `--network host` to share the robot's ROS 2 network, **and outbound HTTPS** to reach the vision service |
| Compute | no GPU required |
| **Safety** | **an operator must hold the emergency stop.** The container asserts this before it moves. |

## What it does each round

1. Parks the idle arm inside the base envelope. Both arms share one rail, so the
   idle one is put away before anything descends.
2. Homes the working arm.
3. Drives until the object is reachable, steering on what it can see.
4. Finds the object, picks it, carries it, and places it on the named
   destination.
5. Returns the arm to its home position at the end, whatever happened.

Progress is printed as it goes.

## Check before a run

```bash
docker run --rm ebim-task3 --selftest   # loads the policy and exits; moves nothing
docker run --rm ebim-task3 --version    # build stamp
```

## Exit codes

| code | meaning | what to check |
|---|---|---|
| `0` | completed | — |
| `3` | the image is incomplete | contact us |
| `4` | vision service unset or unreachable | `VISION_ENDPOINT`, and outbound HTTPS from the host |
| `5` | no robot found on the ROS network | the control stack, `ROS_DOMAIN_ID`, and `--network host` |
| other | a stage failed | the last lines of output name the stage |

If a run fails, the printed output is the diagnosis. Please send the last 20
lines and the output of `--version`.

## Between rounds

Return the robot to its marked start pose, put the items back on the kitchen
table, clear the destination surface, and open the gripper. Each `docker run`
starts a fresh round; nothing carries over.

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to publish
the result. No other use is permitted.
