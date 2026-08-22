# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""R9 T1 -- the VM A / VM B grasp interface (plans/LOOP_PROMPT_VM_A_REV9.md
"The contract"). VM B is blocked on this landing.

Three artifacts, three directions of data flow:

1. **Candidates** (VM B writes, VM A reads) --
   `assets/derived/grasp_candidates/<object>.json`. ER-2 (or any future
   candidate source) proposes N grasp poses per object.
2. **Ranked** (VM A writes, VM B reads) --
   `assets/derived/grasp_ranked/<object>.json`. The real Lula IK-feasibility
   sweep (`scripts/task3/ik_feasibility_sweep.py` rank mode) scores every
   candidate x arm side x base stance and orders the feasible ones.
3. **Grasp memory** (both append) -- `assets/derived/grasp_memory.jsonl`,
   one line per real GPU attempt: predicted-vs-actual outcome. Append-only
   by construction (one O_APPEND write per call) so two machines writing
   concurrently never corrupt each other's entries.

Deliberately small: `position`, `yaw_rad`, `tilt_rad`, `side`, `stance` are
the fields that matter (rev 9's own instruction -- do not gold-plate).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GRASP_CANDIDATES_DIR = REPO_ROOT / "assets" / "derived" / "grasp_candidates"
GRASP_RANKED_DIR = REPO_ROOT / "assets" / "derived" / "grasp_ranked"
GRASP_MEMORY_PATH = REPO_ROOT / "assets" / "derived" / "grasp_memory.jsonl"

VALID_SIDES = ("left", "right")


class GraspContractError(ValueError):
    """A candidates/ranked/memory artifact does not match the contract."""


def candidates_path(
    object_name: str, base_dir: Path = GRASP_CANDIDATES_DIR
) -> Path:
    return base_dir / f"{object_name}.json"


def ranked_path(object_name: str, base_dir: Path = GRASP_RANKED_DIR) -> Path:
    return base_dir / f"{object_name}.json"


def _require(raw: Any, fields: tuple[str, ...], what: str) -> None:
    if not isinstance(raw, dict):
        raise GraspContractError(
            f"{what} must be a JSON object, got {type(raw).__name__}"
        )
    missing = [f for f in fields if f not in raw]
    if missing:
        raise GraspContractError(
            f"{what} missing required field(s): {missing}"
        )


def _require_xyz(raw: Any, what: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise GraspContractError(f"{what} must be [x, y, z]")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _require_side(raw: Any, what: str) -> str:
    side = str(raw)
    if side not in VALID_SIDES:
        raise GraspContractError(
            f"{what} must be one of {VALID_SIDES}, got {side!r}"
        )
    return side


@dataclass(frozen=True)
class GraspCandidate:
    """One proposed grasp pose for an object (VM B / ER-2's output unit)."""

    id: int
    position: tuple[float, float, float]
    yaw_rad: float
    tilt_rad: float
    source: str
    label: str
    confidence: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": list(self.position),
            "yaw_rad": self.yaw_rad,
            "tilt_rad": self.tilt_rad,
            "source": self.source,
            "label": self.label,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> GraspCandidate:
        _require(
            raw,
            (
                "id",
                "position",
                "yaw_rad",
                "tilt_rad",
                "source",
                "label",
                "confidence",
            ),
            "candidate",
        )
        position = _require_xyz(raw["position"], "candidate.position")
        return GraspCandidate(
            id=int(raw["id"]),
            position=position,
            yaw_rad=float(raw["yaw_rad"]),
            tilt_rad=float(raw["tilt_rad"]),
            source=str(raw["source"]),
            label=str(raw["label"]),
            confidence=float(raw["confidence"]),
        )


@dataclass(frozen=True)
class CandidateFile:
    object: str
    object_pose: tuple[float, float, float]
    generated_utc: str
    candidates: tuple[GraspCandidate, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "object_pose": list(self.object_pose),
            "generated_utc": self.generated_utc,
            "candidates": [c.to_json() for c in self.candidates],
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> CandidateFile:
        _require(
            raw,
            ("object", "object_pose", "generated_utc", "candidates"),
            "candidate file",
        )
        object_pose = _require_xyz(raw["object_pose"], "object_pose")
        if not isinstance(raw["candidates"], list) or not raw["candidates"]:
            raise GraspContractError("candidates must be a non-empty list")
        candidates = tuple(
            GraspCandidate.from_json(c) for c in raw["candidates"]
        )
        ids = [c.id for c in candidates]
        if len(ids) != len(set(ids)):
            raise GraspContractError(f"duplicate candidate ids: {ids}")
        return CandidateFile(
            object=str(raw["object"]),
            object_pose=object_pose,
            generated_utc=str(raw["generated_utc"]),
            candidates=candidates,
        )


@dataclass(frozen=True)
class RankedGrasp:
    """One (candidate, arm side, base stance) combination, IK-scored."""

    candidate_id: int
    side: str
    stance_xy: tuple[float, float]
    stance_yaw: float
    feasible: bool
    ik_margin: float
    rank: int

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "side": self.side,
            "stance_xy": list(self.stance_xy),
            "stance_yaw": self.stance_yaw,
            "feasible": self.feasible,
            "ik_margin": self.ik_margin,
            "rank": self.rank,
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> RankedGrasp:
        _require(
            raw,
            (
                "candidate_id",
                "side",
                "stance_xy",
                "stance_yaw",
                "feasible",
                "ik_margin",
                "rank",
            ),
            "ranked entry",
        )
        side = _require_side(raw["side"], "ranked entry.side")
        stance_xy = raw["stance_xy"]
        if not isinstance(stance_xy, (list, tuple)) or len(stance_xy) != 2:
            raise GraspContractError("stance_xy must be [x, y]")
        return RankedGrasp(
            candidate_id=int(raw["candidate_id"]),
            side=side,
            stance_xy=(float(stance_xy[0]), float(stance_xy[1])),
            stance_yaw=float(raw["stance_yaw"]),
            feasible=bool(raw["feasible"]),
            ik_margin=float(raw["ik_margin"]),
            rank=int(raw["rank"]),
        )


@dataclass(frozen=True)
class RankedFile:
    object: str
    ranked: tuple[RankedGrasp, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "ranked": [r.to_json() for r in self.ranked],
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> RankedFile:
        _require(raw, ("object", "ranked"), "ranked file")
        if not isinstance(raw["ranked"], list):
            raise GraspContractError("ranked must be a list")
        ranked = tuple(RankedGrasp.from_json(r) for r in raw["ranked"])
        ranks = [r.rank for r in ranked]
        if ranks != sorted(ranks):
            raise GraspContractError(
                f"ranked entries must be sorted by rank ascending, got {ranks}"
            )
        return RankedFile(object=str(raw["object"]), ranked=ranked)

    def feasible_sorted(self) -> tuple[RankedGrasp, ...]:
        """Feasible entries in rank order -- what the live grasp path
        (T4) walks, trying each until one succeeds or the attempt cap is
        hit."""
        return tuple(r for r in self.ranked if r.feasible)


@dataclass(frozen=True)
class GraspMemoryEntry:
    """One real attempt, predicted vs. actual -- the learning-loop record
    (rev 9's stated purpose: seed next episode's ranking from what actually
    worked, and double as a labelled dataset for a future VLA fine-tune)."""

    object: str
    candidate_id: int
    side: str
    stance_xy: tuple[float, float]
    stance_yaw: float
    predicted_feasible: bool
    object_follows_ee: bool
    position_error_m: float
    utc: str
    source: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "candidate_id": self.candidate_id,
            "side": self.side,
            "stance_xy": list(self.stance_xy),
            "stance_yaw": self.stance_yaw,
            "predicted_feasible": self.predicted_feasible,
            "object_follows_ee": self.object_follows_ee,
            "position_error_m": self.position_error_m,
            "utc": self.utc,
            "source": self.source,
        }

    @staticmethod
    def from_json(raw: dict[str, Any]) -> GraspMemoryEntry:
        _require(
            raw,
            (
                "object",
                "candidate_id",
                "side",
                "stance_xy",
                "stance_yaw",
                "predicted_feasible",
                "object_follows_ee",
                "position_error_m",
                "utc",
            ),
            "memory entry",
        )
        side = _require_side(raw["side"], "memory entry.side")
        stance_xy = raw["stance_xy"]
        if not isinstance(stance_xy, (list, tuple)) or len(stance_xy) != 2:
            raise GraspContractError("stance_xy must be [x, y]")
        return GraspMemoryEntry(
            object=str(raw["object"]),
            candidate_id=int(raw["candidate_id"]),
            side=side,
            stance_xy=(float(stance_xy[0]), float(stance_xy[1])),
            stance_yaw=float(raw["stance_yaw"]),
            predicted_feasible=bool(raw["predicted_feasible"]),
            object_follows_ee=bool(raw["object_follows_ee"]),
            position_error_m=float(raw["position_error_m"]),
            utc=str(raw["utc"]),
            source=str(raw.get("source", "")),
        )


def load_candidates(
    object_name: str, base_dir: Path = GRASP_CANDIDATES_DIR
) -> CandidateFile:
    path = candidates_path(object_name, base_dir)
    raw = json.loads(path.read_text())
    candidate_file = CandidateFile.from_json(raw)
    if candidate_file.object != object_name:
        raise GraspContractError(
            f"{path}: object field {candidate_file.object!r} != "
            f"filename {object_name!r}"
        )
    return candidate_file


def save_ranked(
    ranked_file: RankedFile, base_dir: Path = GRASP_RANKED_DIR
) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = ranked_path(ranked_file.object, base_dir)
    path.write_text(
        json.dumps(ranked_file.to_json(), indent=2, sort_keys=True)
    )
    return path


def save_candidates(
    candidate_file: CandidateFile, base_dir: Path = GRASP_CANDIDATES_DIR
) -> Path:
    """REV12 T4: persists a candidate file that may include synthetic
    proven-height candidates (task3_autonomy.grasp_ranking.
    inject_proven_candidates) alongside any real ones -- so a later
    `load_candidates` (e.g. `_load_ranked_grasp_plan`'s `by_id` lookup)
    can resolve every candidate_id a ranked file references, not just the
    ones a candidate source originally wrote."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = candidates_path(candidate_file.object, base_dir)
    path.write_text(
        json.dumps(candidate_file.to_json(), indent=2, sort_keys=True)
    )
    return path


def load_ranked(
    object_name: str, base_dir: Path = GRASP_RANKED_DIR
) -> RankedFile:
    path = ranked_path(object_name, base_dir)
    raw = json.loads(path.read_text())
    ranked_file = RankedFile.from_json(raw)
    if ranked_file.object != object_name:
        raise GraspContractError(
            f"{path}: object field {ranked_file.object!r} != "
            f"filename {object_name!r}"
        )
    return ranked_file


def append_grasp_memory(
    entry: GraspMemoryEntry, path: Path = GRASP_MEMORY_PATH
) -> None:
    """One O_APPEND write of one JSON line -- safe for VM A and VM B to
    call concurrently against the same file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry.to_json(), sort_keys=True)
    with path.open("a") as fh:
        fh.write(line + "\n")


def read_grasp_memory(
    path: Path = GRASP_MEMORY_PATH,
) -> Iterator[GraspMemoryEntry]:
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield GraspMemoryEntry.from_json(json.loads(line))
