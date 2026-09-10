# EBiM Task 3 — Assisted Living & Feeding

Autonomous table setting and clearing on a dual-arm mobile manipulator.
Everything needed to run it is on this page.

## 1 · The image

```
ghcr.io/sushruths04/ebim-task3-mission:1.3.0
```

Built for **linux/amd64 and linux/arm64**. The companion is a Jetson (arm64) and
Docker selects the right build automatically — nothing is emulated on the robot.

Nothing to build. Pull it, or let `docker run` pull it on first use.

## 2 · Which machine to run it on

**The computer that runs the FR3 arm control stack** — the one where this prints
topics:

```bash
ros2 topic list | grep /right/
```

This image is ROS 2 Humble, and ROS 2 does not communicate across distros. On a
host running a different distro it reports `no /right/* topics` and stops, while
the container and the robot are both perfectly healthy.

## 3 · What must already be running

- the FR3 arm control stack, accepting goals
- the cameras publishing
- the mobile base
- **an operator holding the emergency stop** — the container refuses to move
  without `--i-am-on-the-estop`, and returns the arm home at the end whatever
  happens

## 4 · Checks that move nothing

Safe at any time, a few seconds each. Please run these first.

```bash
# a) is the image complete?
docker run --rm ghcr.io/sushruths04/ebim-task3-mission:1.3.0 --selftest

# b) print the whole driving route with clearances
docker run --rm --network host -e ROS_DOMAIN_ID=0 \
  ghcr.io/sushruths04/ebim-task3-mission:1.3.0 --plan

# c) on the robot: is it localised in the map?
ros2 run tf2_ros tf2_echo map base_link
```

- **(a)** should end `self-test PASSED`.
- **(b)** should end `16/16 legs clear. ROUTE IS DRIVEABLE.`
- **(c)** is the important one. The autonomous stages drive to coordinates in a
  map, so the robot must place itself in that map for them to mean anything. If
  this prints a transform, the navigation is usable here. **If it cannot find
  the frame, please tell us and run only the pick-and-place** (section 5).

## 5 · Scenario A — a single pick-and-place

The robot picks one named object and puts it on another, both in front of it.
**It does not drive between rooms.** Put the cup and the plate on the table in
front of the robot.

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=<supplied to you directly> \
  -e TMR_WS=/ws -v /home/tmr-user/ros2_ws:/ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.3.0 \
  autorun --object "the rim of the cup" --onto "the rim of the plate" \
  --i-am-on-the-estop
```

What it does, printed as it goes:

1. Parks the left arm inside the base envelope — both arms share one rail, so
   the idle one is put away before anything descends.
2. Homes the right arm.
3. Drives the base in small steps until the object is reachable, searching back,
   then left, then right if neither camera can see it. Gives up bounded rather
   than wandering.
4. Locates the object and the destination in one look, in the arm's own frame.
5. Picks, returns through home, places, releases.
6. Homes the arm.

Any pair can be named, e.g. `--object "the rim of the bowl" --onto "the rim of
the plate"`.

## 6 · Scenario B — the autonomous stages

### Stage 1 — kitchen to the dining table

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=<supplied to you directly> \
  -e TMR_WS=/ws -v /home/tmr-user/ros2_ws:/ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.3.0 \
  --stage 1 --i-am-on-the-estop
```

Starting from its dock, for the cup, then the bowl, then the plate:

| step | what happens |
|---|---|
| **0** | Parks the left arm and homes the right one. Once, at the start. |
| **1** | Plans a route around the map and whatever the sensors see now, then drives to the kitchen table. A leg with less than the robot's own width of clearance is **refused, not driven**. |
| **2** | At the table, closes the last stretch on the **object**, not on the waypoint — steering on what the cameras report, searching back / left / right if it is not yet in view. |
| **3** | Picks it. |
| **4** | **Raises the rail before driving.** The pick leaves the arm at the height of the table it took from; driving a room's length in that posture would carry the object at table height past everything on the route. |
| **5** | Drives to the dining table, to the pose facing that item's lettered place. |
| **6** | Finds the letter with the head camera and releases the object there. If the letter cannot be seen it places at the table edge the waypoint guarantees, and says so in the log. |
| **7** | Returns to the kitchen for the next item. |

The arm is homed at the end regardless of outcome.

### Stage 4 — dining table back to the kitchen

```bash
... ghcr.io/sushruths04/ebim-task3-mission:1.3.0 --stage 4 --i-am-on-the-estop
```

The same motion in the other direction: each item is collected from its lettered
place on the dining table and returned to the marked area in the kitchen.

`--stage all` runs Stage 1 and then Stage 4.

## 7 · Every command

| command | moves the robot? | what it does |
|---|---|---|
| `--selftest` | no | the image is complete |
| `--plan` | no | prints the whole route with clearances |
| `--version` | no | build stamp |
| `autorun --object X --onto Y --i-am-on-the-estop` | **yes** | one pick-and-place |
| `--stage 1 --i-am-on-the-estop` | **yes** | kitchen → dining table |
| `--stage 4 --i-am-on-the-estop` | **yes** | dining table → kitchen |
| `--stage all --i-am-on-the-estop` | **yes** | both, in order |
| `--fresh` | — | ignore a part-finished run and start over |

**Without `--i-am-on-the-estop` nothing is commanded**, whatever else is passed.

## 8 · Environment

| variable | required | meaning |
|---|---|---|
| `VISION_ENDPOINT` | for a real run | the vision service, supplied to you directly. Hosted and operated by us — nothing to set up, no credential for you to manage. Not needed for `--plan` or `--selftest`. |
| `TMR_WS` + the `-v` mount | **yes, for a real run** | the robot's own ROS workspace. The policy needs `franka_msgs` from it; the packaged version lacks the `PTPMotion` action. |
| `ROS_DOMAIN_ID` | no | must match the control stack; default `0` |

## 8b · Hosting perception yourself

The policy asks one thing of perception: an image and a phrase in, one pixel
out. No calibration, no knowledge of the robot. Any service answering that
contract can be substituted — including one you host.
See [ALTERNATIVES.md](ALTERNATIVES.md) for the contract and for open models that
fit it unmodified.

## 9 · If something goes wrong

| exit code | meaning | what to check |
|---|---|---|
| `0` | finished | — |
| `1` | a stage failed, or the route is not driveable | the output names which |
| `3` | the image is incomplete | contact us |
| `4` | vision service unset or unreachable | `VISION_ENDPOINT`, and outbound HTTPS from that machine |
| `5` | no robot found | the control stack, `ROS_DOMAIN_ID`, `--network host`, and that you are on the arm computer |

Two messages are **refusals, not faults** — the policy declining to do something
unsafe, with nothing moved:

- *"a leg has less clearance than the robot's half-width"* — the route is
  blocked and it declined to drive it.
- *"joint N would move X rad, over the cap"* — an arm goal was rejected before
  being sent. The object is still held.

Every stage prints its own measurement, and a run stops at the first stage that
fails rather than carrying a fault forward. **If a run fails, those printed
lines are the diagnosis** — please send the last 20 lines and the output of
`--version`.

## 10 · Between rounds

Return the robot to its marked start pose, put the items back on the kitchen
table, clear the destination surface, and open the gripper. Each `docker run`
starts a fresh round; nothing carries over.

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to publish
the result. No other use is permitted.
