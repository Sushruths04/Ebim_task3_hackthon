# Operating guide — EBiM Task 3

Everything needed to run this on the testbed. Two things can be run: a **single
pick-and-place**, and the **full autonomous Stage 1**. Both come from one image.

---

## 0 · Before anything

**One image, both architectures.** The companion is arm64; Docker selects the
right build automatically.

```
ghcr.io/sushruths04/ebim-task3-mission:1.4.0
```

**On the Jetson, every `docker` command needs `--network host`** — its kernel
lacks a module Docker's default networking needs. Every command below has it.

**Which machine.** The computer that runs the FR3 arm control stack — the one
where this prints topics:

```bash
ros2 topic list | grep /right/
```

This image is ROS 2 Humble. ROS 2 does not communicate across distros, so on a
host running a different distro it will report `no /right/* topics` and stop,
while the container and the robot are both perfectly healthy.

**What must be running already:** the arm control stack accepting goals, the
cameras publishing, and the mobile base. **An operator must be holding the
emergency stop** — the container refuses to move without `--i-am-on-the-estop`.

**Three things every real run needs:**

| | |
|---|---|
| `export VISION_ENDPOINT='...'` once, then `-e VISION_ENDPOINT` | the vision service, supplied to you directly |
| `-v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro` | the robot's arm workspace. The policy needs `franka_msgs` from it — the packaged version lacks the `PTPMotion` action. |
| `-v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro` | the robot's spine workspace, for `franka_spine_msgs` |

**Mount each workspace at its own path**, exactly as above. They are built with
symlinks to absolute paths, so mounted anywhere else they appear empty; the
container detects that and prints the path to use.

The robot's home pose is carried in the image. To use the host's current file
instead, add
`-v /home/tmr-user/teleop_home_pose.yaml:/home/tmr-user/teleop_home_pose.yaml:ro`;
the log says which one is in use.

---

## 1 · Check the image (moves nothing, ~20 s)

```bash
docker run --rm --network host \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.0 --selftest
```

Expect `self-test PASSED -- image complete, robot workspaces verified`. This
builds every robot client and reads every calibration file exactly as a real
run does, and plans the route against the map. Nothing is commanded.

---

## 2 · Scenario A — a single pick-and-place

**What it does.** The robot picks one named object and puts it on another named
destination, both in front of it. **It does not drive between rooms.**

**Set up:** put the cup and the plate on the table in front of the robot.

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.0 \
  autorun --object "the rim of the cup" --onto "the rim of the plate" \
  --i-am-on-the-estop
```

**Sequence, printed as it goes:**

1. Parks the left arm inside the base envelope. Both arms share one rail, so the
   idle one is put away before anything descends.
2. Homes the right arm.
3. Drives the base in small steps until the object is reachable, searching back,
   then left, then right if neither camera can see it. Gives up bounded rather
   than wandering.
4. Locates the object and the destination in one look, in the arm's own frame.
5. Picks, returns through home, places, releases.
6. Homes the arm — whatever happened.

Any object and destination can be named: `--object "the rim of the bowl"
--onto "the rim of the plate"`.

---

## 3 · Scenario B — the full autonomous Stage 1

**What it does.** The robot starts at its dock and delivers each item to the
dining table without further input.

### 3.1 Test run first (moves nothing, ~10 s)

```bash
docker run --rm --network host -e ROS_DOMAIN_ID=0 \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.0 --plan
```

Prints every drive with its distance and the tightest clearance on it. `16/16
legs clear` means the route is geometrically sound.

### 3.2 Confirm the robot knows where it is

```bash
ros2 run tf2_ros tf2_echo map base_link
```

**This is the one that matters.** The route drives to coordinates in a map. If
this prints a transform, the robot is localised and those coordinates are
meaningful. If it cannot find the frame, localisation is not running and the
navigation cannot be used in this cell — please tell us rather than continuing.

### 3.3 Run it

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.0 \
  --stage 1 --i-am-on-the-estop
```

### 3.4 What the robot does, per item

Starting at the dock, for the cup, then the bowl, then the plate:

| step | what happens |
|---|---|
| **0** | Parks the left arm; homes the right arm. Once, at the start. |
| **1** | Plans a route around the mapped walls and furniture — through the doorway's centre line between rooms — and drives exactly that route to the kitchen table. **A route with less than the robot's own half-width of clearance is refused, not driven.** |
| **2** | At the table, closes the last stretch on the **object**, not on the waypoint — steering on what the cameras report, searching back/left/right if the object is not yet in view. |
| **3** | Picks it. |
| **4** | **Brings the arm home, raises the rail, then drives.** The pick leaves the arm low over the table it took the object from; nothing is carried in that posture. |
| **5** | Drives to the dining table, to the pose facing that item's lettered place. |
| **6** | Looks for the letter with the head camera, releases the object there, and homes the arm. If the letter cannot be seen, it places in front of the pose that faces that letter, and says so in the log. |
| **7** | Drives back to the kitchen for the next item. |

At the end the arm is returned home regardless of outcome.

**Which letter is whose.** By default the cup goes to **c**, the bowl to **b**
and the plate to **a**. If the round assigns them differently, add `--assign`,
for example `--stage 1 --assign cup=b --i-am-on-the-estop`. Naming one object is
enough — the others keep their letter if it is free and otherwise take the one
left over (`cup=b` gives cup b, bowl c, plate a); the run prints the result
first. Pass the same `--assign` to Stage 4, so each item is collected from where
it was put.

**One object, one letter.** `--move` moves only what it names:
`--stage 1 --move cup=a --i-am-on-the-estop` takes the cup from the kitchen
table and places it on **a**, and nothing else;
`--stage 4 --move cup=a --i-am-on-the-estop` brings the cup back from **a** to
the black rectangle. Several at once: `--move cup=a,plate=c`.

### 3.5 Stage 4 — bringing them back

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.0 \
  --stage 4 --i-am-on-the-estop
```

The same motion in the other direction: each item is collected from the dining
table and placed on the black rectangle on the left of the kitchen table.

`--stage all` runs Stage 1 then Stage 4.

---

## 4 · Every command, in one place

| command | moves the robot? | what it does |
|---|---|---|
| `--selftest` | no | image is complete; with the workspaces mounted, verifies them too |
| `--plan` | no | prints the whole route with clearances |
| `--version` | no | build stamp |
| `autorun --object X --onto Y --i-am-on-the-estop` | **yes** | one pick-and-place |
| `--stage 1 --i-am-on-the-estop` | **yes** | kitchen → dining table |
| `--stage 4 --i-am-on-the-estop` | **yes** | dining table → kitchen |
| `--stage all --i-am-on-the-estop` | **yes** | both, in order |
| `--fresh` | — | ignore a part-finished run and start over |
| `--assign cup=b` | — | which lettered place each object goes to; default `cup=c,bowl=b,plate=a` |
| `--move cup=a` | — | move only the named object(s): stage 1 to that letter, stage 4 back from it |

Without `--i-am-on-the-estop` nothing is commanded, whatever else is passed.

---

## 5 · If something goes wrong

Before anything moves, a real run checks the robot's messages, the vision
service and the robot, in that order, and stops at the first that fails.

| exit code | meaning | what to check |
|---|---|---|
| `0` | finished | — |
| `1` | a stage failed, or the route is not driveable | the output names which |
| `3` | the image is incomplete | contact us |
| `4` | vision service unset, or no answer within 45 s | `VISION_ENDPOINT`, and outbound HTTPS from that machine |
| `5` | no robot found | the control stack, `ROS_DOMAIN_ID`, `--network host`, and that you are on the arm computer |
| `6` | the robot workspaces are not visible | the two workspace mounts, each at its own path (section 0) |

Every stage prints its own measurement as it runs, and the run stops at the
first stage that fails rather than carrying a fault forward. **If a run fails,
those printed lines are the diagnosis** — please send the last 20 lines and the
output of `--version`.

Two messages that are refusals rather than faults, and mean the policy is
working as intended:

- **"a leg has less clearance than the robot's half-width"** — the route is
  blocked; the robot declined to drive it.
- **"joint N would move X rad, over the cap"** — an arm goal was rejected before
  it was sent. The object is still held and nothing has moved.

---

## 6 · Between rounds

Return the robot to its marked start pose, put the items back on the kitchen
table, clear the destination surface, and open the gripper. Each `docker run`
starts a fresh round; nothing carries over.
