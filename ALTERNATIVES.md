# The perception backend is replaceable

The policy does not depend on any particular vision service. This page shows how
to point it at a different one — including one you host yourself — and gives
measured numbers for several open models that fit without modification.

## The whole interface is one function

```
image (JPEG) + a phrase  ->  one pixel (u, v)
```

That is all the policy asks of perception. Everything that follows — turning
that pixel into a point in the arm's frame, fitting the object's rim, planning
the approach and the grasp — is the policy's own geometry and is unaffected by
what answered.

So a backend needs **no camera calibration, no intrinsics, no hand-eye
transform, and no knowledge of the robot.** It looks at a picture and returns a
coordinate.

## Pointing it somewhere else

Set one variable:

```bash
-e VISION_ENDPOINT=https://your-own-host/
```

The service at that address receives:

```json
{ "image_b64": "<JPEG, base64>", "description": "the rim of the cup" }
```

and must answer:

```json
{ "ok": true, "point_norm": [y, x] }
```

`point_norm` is **normalised 0–1000, y first**; the caller scales it to the
image. An empty `image_b64` is a reachability probe and should return
`{"ok": false, "probe": true}` without doing any work.

`GET /health` must return 200 — the container checks it before it moves.

That is the entire contract. Roughly forty lines of server code around whichever
model you choose.

## Models that fit it, unmodified

Every one of these is instruction-tuned for spatial referring already:
**no fine-tuning, no calibration, no code change.** They differ only in
accuracy and in what they need to run.

| model | licence | needs a GPU? |
|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | Apache-2.0 | yes |
| `BAAI/RoboBrain2.5-4B` | Apache-2.0 | yes |
| `Qwen/Qwen3-VL-2B-Instruct` | Apache-2.0 | yes |
| `nvidia/Cosmos-Reason2-2B` | NVIDIA, gated | yes |
| `sush0401/ebim-bowl-pointer` | ours, 5.6 MB | **no — ~35 ms on CPU** |

### One thing that will silently halve accuracy

These models answer in **relative 0–1000 coordinates, not pixels.** A reply of
`[199, 345]` for a 640×480 image is not a pixel pair:

```python
u = x / 1000 * image_width
v = y / 1000 * image_height
```

Measured on this rig: of twelve answers across three models, eleven preferred
this reading. Treating them as pixels moved one answer from 60 px to 156 px.

## How accurate, honestly

Measured against the default service's own answers on a frame from this robot.
**This is a small sample — one frame, four objects — and it is reported as such.**

| | cup | plate | bowl | spoon |
|---|---|---|---|---|
| `RoboBrain2.5-4B` | 8 px | 88 px | 44 px | 28 px |
| `Qwen3-VL-4B` | 24 px | 15 px | 194 px | 45 px |
| `Cosmos-Reason2-2B` | 16 px | 101 px | 87 px | 54 px |

Read the rows rather than an average. Each model is close on some objects and
poor on others, and not the same ones — that 194 px is a frame containing two
bowls, where the phrase was ambiguous rather than the model wrong.

**These are substitutes, not equivalents.** They exist so the policy can run if
the default service is unavailable, on hardware you control. For a scored run,
the default service is what the numbers in the technical report were obtained
with.

## The CPU option

`sush0401/ebim-bowl-pointer` — 5.6 MB, ~35 ms per frame, no GPU and no network
at all. Distilled from a larger model onto this robot's own recorded grasps.

It detects **the bowl only**: the source demonstrations are bowl grasps, so that
is what the data supports. It declines frames where no bowl is present rather
than guessing, which is the behaviour that matters on a robot.

## Summary

| you want | use |
|---|---|
| the documented configuration | the default service |
| to host perception yourself, GPU available | `Qwen3-VL-4B` or `RoboBrain2.5-4B` behind the contract above |
| no GPU, no network | `ebim-bowl-pointer`, for the bowl |

No option requires changing the policy, recalibrating the robot, or fine-tuning
anything.
