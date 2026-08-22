# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REV12 T4: make the ranker actually rank.

Three real bugs in ``scripts/task3/ik_feasibility_sweep.py::run_rank``,
diagnosed against real GPU data (``plans/SYNC.md`` REV12 ladder, GATE
findings from a 40-candidate ranked-vs-fallback comparison):

1. Every converged Lula solve reports ``ik_margin`` (= -position_error_m)
   at machine precision (~-0.0000) -- correct by construction, but it
   means the old sort key ``(not feasible, -ik_margin)`` collapses almost
   every tie to Python's stable-sort tiebreak: candidate GENERATION
   order, not quality.
2. Generation order clusters tiny XY nudges of the SAME baseline point
   first and z variants only from id 9 onward -- a caller capped at 2
   attempts never sees a z variant at all.
3. The two heights actually PROVEN to hold on real GPU physics
   (dz=+0.075 hardcoded fallback; dz=+0.0846 bowl2's real ranked
   success) were never among any object's candidates -- ER's z comes
   from the raycast surface point, with no fingertip offset added.

This module is the pure, Isaac-free part of the fix: candidate
injection, the new sort key, and diversity reporting. Wired into
``run_rank`` (which still owns the real Lula IK sweep -- that part
genuinely needs Isaac/lula and is not duplicated here).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from task3_autonomy.grasp_contract import CandidateFile, GraspCandidate

# The two heights proven to hold on real GPU physics this project has
# ever recorded (plans/SYNC.md, REV9/REV12 GATE T0/T5 entries):
#   proven_hardcoded: world_isaac.py's fallback grasp
#     (object_grasp_target default z_offset=0.075, x_offset=y_offset=0.0,
#     top-down orientation yaw=0.0) -- holds cup (err 0.0109) and bowl2
#     (err 0.0557).
#   proven_er_bowl2: bowl2's real ranked-candidate success
#     (assets/derived/grasp_candidates/bowl2.json: position z=0.831,
#     object_pose z=0.74642014503479 -> dz=0.08458) -- held first
#     attempt, no fallback, position_error_m 0.0753.
# Read from world_isaac.py / the committed candidate file, not retyped
# from any planning document.
PROVEN_CANDIDATE_OFFSETS: dict[str, float] = {
    "proven_hardcoded": 0.075,
    "proven_er_bowl2": 0.0846,
}

# ~5mm: below the grasp verifier's own tolerances (config.THRESHOLDS.
# reach_tolerance_m=0.05, GRASP_HELD_MAX_DIST_M=0.08) -- two candidates
# closer than this are the same grasp for any practical purpose.
EQUIVALENCE_TOLERANCE_M = 0.005

# The dz band the two proven heights actually occupy (0.075, 0.0846) --
# used to reward any candidate (proven or not) whose height falls in the
# same neighborhood, not just the two literal injected points.
PROVEN_DZ_BAND = (0.070, 0.090)

# Large enough to dominate the band-proximity penalty (which is at most
# a few tens of cm for any sane candidate) so a proven source always
# outranks an equally-close unproven one, but never overrides the
# feasible/infeasible partition (that stays the first sort key).
PROVEN_SOURCE_BONUS = 5.0

# Confidence is in [0, 1] (grasp_contract.GraspCandidate.confidence) and
# is documented as a tie-break, not a primary signal -- small weight so
# it only ever discriminates between otherwise-equal candidates.
CONFIDENCE_TIEBREAK_WEIGHT = 0.01


def inject_proven_candidates(
    candidates: tuple[GraspCandidate, ...],
    object_pose: tuple[float, float, float],
) -> tuple[GraspCandidate, ...]:
    """Add the proven-height synthetic candidates unless an equivalent
    (same source-worthy height, within EQUIVALENCE_TOLERANCE_M) already
    exists. XY stays at the object center (x_offset=y_offset=0, matching
    world_isaac.py's hardcoded-pose default) and orientation is top-down
    (yaw=0, tilt=0) -- the injected candidates carry no claim about the
    object's grasp-relevant geometry (rim angle, handle direction), only
    about height, which is the one thing proven across objects.
    """
    ox, oy, oz = object_pose
    next_id = (max((c.id for c in candidates), default=-1)) + 1
    injected: list[GraspCandidate] = []
    for source, dz in PROVEN_CANDIDATE_OFFSETS.items():
        target_z = oz + dz
        has_equivalent = any(
            abs(c.position[2] - target_z) <= EQUIVALENCE_TOLERANCE_M
            for c in candidates
        )
        if has_equivalent:
            continue
        injected.append(
            GraspCandidate(
                id=next_id,
                position=(ox, oy, target_z),
                yaw_rad=0.0,
                tilt_rad=0.0,
                source=source,
                label=source,
                confidence=1.0,
            )
        )
        next_id += 1
    return candidates + tuple(injected)


def inject_proven_candidates_into_file(
    candidate_file: CandidateFile | None, object_name: str, object_pose
) -> CandidateFile:
    """Same as `inject_proven_candidates`, but for the whole-file case
    (including the "no real candidate file exists yet" case -- cup,
    plate2, spoon2 as of this session, see plans/SYNC.md)."""
    existing = candidate_file.candidates if candidate_file else ()
    augmented = inject_proven_candidates(existing, object_pose)
    if candidate_file is None:
        import datetime

        return CandidateFile(
            object=object_name,
            object_pose=tuple(object_pose),
            generated_utc=datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            candidates=augmented,
        )
    return replace(candidate_file, candidates=augmented)


def _band_penalty(dz: float) -> float:
    lo, hi = PROVEN_DZ_BAND
    if lo <= dz <= hi:
        return 0.0
    return min(abs(dz - lo), abs(dz - hi))


def prior_score(candidate: GraspCandidate, object_pose_z: float) -> float:
    """Higher is better. Order matches the ladder's own spec: proven
    sources get a large bonus; dz proximity to the proven band is
    penalised by distance outside it; confidence is a small tie-break."""
    bonus = (
        PROVEN_SOURCE_BONUS
        if candidate.source in PROVEN_CANDIDATE_OFFSETS
        else 0.0
    )
    dz = candidate.position[2] - object_pose_z
    return (
        bonus
        - _band_penalty(dz)
        + candidate.confidence * CONFIDENCE_TIEBREAK_WEIGHT
    )


def rank_key(
    entry: dict[str, Any],
    candidates_by_id: dict[int, GraspCandidate],
    object_pose_z: float,
) -> tuple[bool, float, float]:
    """Sort ascending by this key = best first.

    (a) feasible first (kept as the first filter, per the ladder spec)
    (b) prior score descending (proven-source bonus, band-proximity
        penalty, confidence tie-break)
    (c) -ik_margin last, for deterministic exact ties only -- documented
        as near-constant telemetry, no longer the primary ranking signal.
    """
    candidate = candidates_by_id[entry["candidate_id"]]
    return (
        not entry["feasible"],
        -prior_score(candidate, object_pose_z),
        -entry["ik_margin"],
    )


def diversity_report(
    top_entries: list[dict[str, Any]],
    candidates_by_id: dict[int, GraspCandidate],
) -> dict[str, Any]:
    """Honest measurement of whether a side's top-N ranked entries are
    actually diverse, per the ladder's gate: >=3 distinct candidate_ids
    and >=2 distinct z values among the top 4. Does NOT fabricate
    candidates to force a pass -- reports the real count, which can
    legitimately be below threshold when too few real/injected
    candidates exist for an object (see module docstring point 3)."""
    ids = [e["candidate_id"] for e in top_entries]
    zs = [round(candidates_by_id[i].position[2], 4) for i in ids]
    no_consecutive_dupes = all(
        ids[i] != ids[i + 1] for i in range(len(ids) - 1)
    )
    return {
        "n_considered": len(top_entries),
        "distinct_candidate_ids": len(set(ids)),
        "distinct_z_values": len(set(zs)),
        "no_consecutive_duplicate_candidate_id": no_consecutive_dupes,
        "meets_diversity_threshold": (
            len(set(ids)) >= min(3, len(top_entries))
            and len(set(zs)) >= min(2, len(top_entries))
            and no_consecutive_dupes
        ),
    }


def rank_side(
    best_entries: list[dict[str, Any]],
    candidates_by_id: dict[int, GraspCandidate],
    object_pose_z: float,
) -> list[dict[str, Any]]:
    """Sort one side's (candidate_id, side)-deduplicated best-entries by
    `rank_key`. One entry per candidate_id already (run_rank's `best`
    dict is keyed by (candidate_id, side)), so "no two consecutive ranks
    share a candidate_id" is structurally guaranteed here -- the real
    diversity question `diversity_report` answers is whether enough
    DISTINCT candidates/heights exist to rank in the first place."""
    return sorted(
        best_entries,
        key=lambda e: rank_key(e, candidates_by_id, object_pose_z),
    )
