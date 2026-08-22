# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared annotated-GIF recorder for real-physics Task 3 probe runs.

Reuses the proven capture pattern from ``scripts/task3/run_stage1_fsm.py``
(``_capture_callback``/``_encode_gif``/``capture_factory``) and
``scripts/task3/record_robot_demo.py``: an ``omni.replicator.core`` camera
plus an ``rgb`` ``AnnotatorRegistry`` annotator pulled with
``rep.orchestrator.step()``, encoded to a GIF via PIL. It deliberately does
NOT use ``rep.writers.get("BasicWriter")`` -- that writer caused a
documented runaway in earlier Task 3 video work.

All Isaac/Replicator/PIL/numpy imports are LAZY (inside methods), so this
module imports cleanly with nothing but the standard library -- no Isaac
Sim, Omniverse, PIL, or numpy required to construct or introspect a
``RunRecorder`` or to unit test its pure helper functions.

Integration snippet (see ``scripts/task3/probe_tray_slide.py`` for the only
runner currently wired to this module -- other Task 3 runners, including
``run_stage1_fsm.py`` and ``record_robot_demo.py``, are intentionally NOT
modified in this change and keep their own inline capture code):

    recorder = RunRecorder(args.output_dir, enabled=args.record_video)
    recorder.setup()  # after sim.reset()/scene.reset()
    recorder.capture(phase_name, lines=[...])  # at each phase log
    recorder.finish("run.gif")  # before the runner exits
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def next_frame_path(frames_dir: Path, index: int) -> Path:
    """Zero-padded ``frames_dir/rgb_XXXX.png`` path for frame ``index``."""
    return frames_dir / f"rgb_{index:04d}.png"


def should_capture(frame_count: int, max_frames: int) -> bool:
    """Whether another frame may be captured given the cap.

    Pure predicate factored out of ``RunRecorder.capture`` so the max-frame
    cap logic is unit-testable without Isaac Sim.
    """
    return frame_count < max_frames


def _overlay(image: Any, label: str, lines: Any = ()) -> Any:
    """Draw ``label`` and ``lines`` onto ``image`` with a legibility panel.

    Mutates and returns the same ``PIL.Image``. ``label`` is drawn larger
    (simulated bold via a double-drawn outline) at the top-left; each
    string in ``lines`` is drawn beneath it in the default font. A
    semi-opaque dark rectangle is drawn behind the text first so it stays
    legible over any background.
    """
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    text_lines = [str(line) for line in lines]
    line_height = 16
    panel_height = line_height * (1 + len(text_lines)) + 12
    panel_width = min(
        image.width,
        max([len(label)] + [len(line) for line in text_lines] + [10]) * 8 + 16,
    )
    draw.rectangle(
        [(0, 0), (panel_width, panel_height)],
        fill=(0, 0, 0, 160),
    )
    # Simulated bold: draw the label offset by one pixel in each direction.
    label_pos = (8, 6)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        draw.text(
            (label_pos[0] + dx, label_pos[1] + dy),
            label,
            font=font,
            fill=(255, 255, 0, 255),
        )
    for row, line in enumerate(text_lines):
        draw.text(
            (8, 6 + line_height * (row + 1)),
            line,
            font=font,
            fill=(255, 255, 255, 255),
        )
    return image


def encode_gif(
    frames_dir: Path, output_path: Path, duration_ms: int = 400
) -> Path:
    """Encode all ``frames_dir/rgb_*.png`` (sorted) into ``output_path``.

    Generalized from ``run_stage1_fsm.py``'s ``_encode_gif``. Raises
    ``RuntimeError`` if no frames are present.
    """
    from PIL import Image

    images = [
        Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE)
        for path in sorted(frames_dir.glob("rgb_*.png"))
    ]
    if not images:
        raise RuntimeError(f"No frames were captured in {frames_dir}")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return output_path


class RunRecorder:
    """Optional annotated-GIF recorder for a single probe run.

    Disabled (``enabled=False``, the default caller pattern when
    ``--record-video`` is absent) makes every method a no-op that never
    raises and never touches the filesystem. Recording must never crash a
    real physics run: every Isaac/Replicator/PIL call in ``setup()`` and
    ``capture()`` is wrapped in ``try/except`` that prints a warning to
    stderr and disables further recording instead of propagating.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        enabled: bool = True,
        width: int = 960,
        height: int = 540,
        # r20 calibration: frame the north-counter tray-slide/edge-pinch zone
        # (tray/lip ~x=-4.2, y=-1.2, z=0.8; robot works from the north at
        # y~=-0.7) -- a first pass to verify from the recorded GIF.
        camera_position: tuple[float, float, float] = (-2.3, -3.1, 1.9),
        look_at: tuple[float, float, float] = (-4.1, -1.15, 0.82),
        max_frames: int = 400,
        gif_duration_ms: int = 400,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.width = width
        self.height = height
        self.camera_position = camera_position
        self.look_at = look_at
        self.max_frames = max_frames
        self.gif_duration_ms = gif_duration_ms
        self.frames_dir = self.output_dir / "frames"
        self._frame_count = 0
        self._annotator: Any = None
        self._render_product: Any = None

    def setup(self) -> None:
        """Create the camera/render-product/annotator and prep frames dir.

        No-op if ``enabled`` is False. Never raises: any setup failure
        (e.g. Isaac not present, cameras disabled) prints a warning to
        stderr and disables recording for the rest of the run.
        """
        if not self.enabled:
            return
        try:
            import omni.replicator.core as rep

            self.frames_dir.mkdir(parents=True, exist_ok=True)
            for stale in self.frames_dir.glob("rgb_*.png"):
                stale.unlink()
            self._frame_count = 0
            camera = rep.create.camera(
                position=self.camera_position, look_at=self.look_at
            )
            self._render_product = rep.create.render_product(
                camera, (self.width, self.height)
            )
            self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._annotator.attach([self._render_product])
        except Exception as exc:  # noqa: BLE001
            print(
                f"RunRecorder.setup: recording disabled ({exc})",
                file=sys.stderr,
            )
            self.enabled = False

    def capture(self, label: str, lines: Any = ()) -> None:
        """Pull and save one annotated frame, if enabled and under budget.

        No-op if disabled or the max-frame cap has been reached. Never
        raises.
        """
        if not self.enabled or self._annotator is None:
            return
        if not should_capture(self._frame_count, self.max_frames):
            return
        try:
            import numpy as np
            from PIL import Image

            # Pull-based read of the LATEST rendered RGB frame -- do NOT call
            # rep.orchestrator.step() here. The proven working capture on this
            # VM (run_episode.py's _save_rgb_frame, which made the 160-frame
            # episode.gif) reads the annotator directly: with enable_cameras
            # and an attached render_product, the runner's own per-step
            # sim.step() renders each frame, so get_data() holds fresh pixels.
            # Calling orchestrator.step() here instead returned an EMPTY array
            # in the isaac-lab-2.3.2 container, so every frame was silently
            # skipped by the size guard below (0 frames after 50 min).
            data = np.asarray(self._annotator.get_data())
            # First render or two after attach can be empty/incomplete -- skip
            # (retried at the next capture) rather than crash. Same validity
            # check as the proven run_episode.py path.
            if data.ndim != 3 or data.shape[-1] < 3 or data.size == 0:
                return
            if data.shape[-1] == 4:
                data = data[..., :3]
            image = Image.fromarray(data.astype(np.uint8))
            _overlay(image, label, lines)
            frame_path = next_frame_path(self.frames_dir, self._frame_count)
            image.save(frame_path)
            self._frame_count += 1
        except Exception as exc:  # noqa: BLE001
            print(
                f"RunRecorder.capture: skipped a frame ({exc})",
                file=sys.stderr,
            )

    def finish(self, gif_name: str = "run.gif") -> Path | None:
        """Encode captured frames into ``output_dir/gif_name``.

        No-op (returns ``None``) if disabled or no frames were captured.
        Detaches the annotator and destroys the render product in
        suppressed ``try/except`` blocks. Never raises.
        """
        if not self.enabled or self._frame_count == 0:
            return None
        gif_path: Path | None = None
        try:
            gif_path = encode_gif(
                self.frames_dir,
                self.output_dir / gif_name,
                duration_ms=self.gif_duration_ms,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"RunRecorder.finish: gif encode failed ({exc})",
                file=sys.stderr,
            )
        try:
            if self._annotator is not None:
                self._annotator.detach()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._render_product is not None:
                self._render_product.destroy()
        except Exception:  # noqa: BLE001
            pass
        return gif_path
