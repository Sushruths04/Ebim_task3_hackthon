# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Parsing and frame conversion for the live ER-2 grasp answer.

No network and no GPU: the point of splitting `parse_grasp_answer` and
`grasp_pose_from_answer` out of the live call is that the parts that can be
silently wrong are the parts that can be tested cheaply.
"""

from __future__ import annotations

import math

import pytest

from task3_autonomy.live_er_grasp import (
    ErGraspAnswer,
    azimuth_from_world_points,
    grasp_pose_from_answer,
    parse_grasp_answer,
)

WIDTH, HEIGHT = 640, 480


def test_point_order_is_y_x_not_x_y():
    """Gemini answers [y, x]; the back-projection wants (u, v).

    Getting this backwards yields a valid-looking point on the wrong axis --
    the single most likely silent error in this module, so it is pinned with
    an asymmetric point that cannot pass by coincidence.
    """
    answer = parse_grasp_answer(
        {
            "grasp_point": [250, 750],  # y=250 -> v=120, x=750 -> u=480
            "approach_from": [250, 750],
            "tilt_deg": 0,
            "roll_deg": 0,
        },
        WIDTH,
        HEIGHT,
    )
    assert answer.grasp_uv == pytest.approx((480.0, 120.0))


def test_parses_a_full_answer():
    answer = parse_grasp_answer(
        {
            "grasp_point": [500, 500],
            "approach_from": [500, 100],
            "tilt_deg": 62.0,
            "roll_deg": -30.0,
            "reason": "rim",
        },
        WIDTH,
        HEIGHT,
    )
    assert answer.tilt_deg == pytest.approx(62.0)
    assert answer.roll_deg == pytest.approx(-30.0)
    assert answer.reason == "rim"


def test_accepts_a_response_wrapped_in_an_array():
    """Observed model behaviour: it sometimes returns [ {...} ]."""
    answer = parse_grasp_answer(
        [{"grasp_point": [1, 2], "approach_from": [3, 4], "tilt_deg": 0, "roll_deg": 0}],
        WIDTH,
        HEIGHT,
    )
    assert answer.grasp_uv[0] == pytest.approx(2 / 1000 * WIDTH)


def test_accepts_a_json_string():
    answer = parse_grasp_answer(
        '{"grasp_point":[10,20],"approach_from":[30,40],"tilt_deg":5,"roll_deg":0}',
        WIDTH,
        HEIGHT,
    )
    assert answer.tilt_deg == pytest.approx(5.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"grasp_point": [500, 1200], "approach_from": [0, 0], "tilt_deg": 0, "roll_deg": 0},
        {"grasp_point": [-5, 500], "approach_from": [0, 0], "tilt_deg": 0, "roll_deg": 0},
        {"grasp_point": [500], "approach_from": [0, 0], "tilt_deg": 0, "roll_deg": 0},
        {"approach_from": [0, 0], "tilt_deg": 0, "roll_deg": 0},
        {"grasp_point": "middle", "approach_from": [0, 0], "tilt_deg": 0, "roll_deg": 0},
        "not json at all",
    ],
)
def test_rejects_malformed_answers_rather_than_clamping(bad):
    """An out-of-frame or missing point is a wrong answer, not a near miss.

    Clamping would hand `reach()` a confident grasp point on the image
    border; raising makes the caller fall back to the old path instead.
    """
    with pytest.raises(ValueError):
        parse_grasp_answer(bad, WIDTH, HEIGHT)


def test_tilt_beyond_ninety_is_clamped_not_rejected():
    """A model that says 105 means "as far from the side as possible".

    Unlike a bad pixel this one has an unambiguous intended meaning, so it is
    held at the limit rather than discarding an otherwise usable answer.
    """
    answer = parse_grasp_answer(
        {"grasp_point": [1, 1], "approach_from": [2, 2], "tilt_deg": 105.0, "roll_deg": 0},
        WIDTH,
        HEIGHT,
    )
    assert answer.tilt_deg == pytest.approx(90.0)


def test_azimuth_points_from_the_approach_side_toward_the_object():
    """The approach axis travels FROM where the gripper starts TO the object."""
    assert azimuth_from_world_points((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert azimuth_from_world_points((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(-90.0)


def test_azimuth_is_none_when_the_approach_is_purely_vertical():
    """No horizontal separation means ER-2 described a top-down grasp."""
    assert azimuth_from_world_points((1.0, 2.0, 3.0), (1.0, 2.0, 9.0)) is None


def _identity_camera():
    """View/projection for a camera at the world origin looking down -Z.

    Matches `perception_grasp.project_to_world`'s contract, which is a
    row-vector convention: the view matrix is identity here, so camera space
    IS world space, which makes the expected world coordinates readable by
    inspection instead of requiring a second implementation to check them.
    proj[0][0]/proj[1][1] are the NDC focal terms.
    """
    view = [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
    proj = [[2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -1.0],
            [0.0, 0.0, -0.2, 0.0]]
    return view, proj


def test_a_vertical_image_offset_collapses_to_a_top_down_grasp():
    """`approach_from` directly above the grasp point in a camera whose image
    'up' is world +Z... is not this camera. With an identity camera, image
    'up' is world -Y, so this instead checks the None-azimuth branch is only
    taken on genuine coincidence."""
    view, proj = _identity_camera()
    answer = ErGraspAnswer(
        grasp_uv=(320.0, 240.0), approach_uv=(320.0, 240.0), tilt_deg=70.0, roll_deg=0.0
    )
    pose = grasp_pose_from_answer(answer, 1.0, view, proj, WIDTH, HEIGHT)
    assert pose.tilt_deg == pytest.approx(0.0)
    assert pose.azimuth_deg == pytest.approx(0.0)


def test_world_pose_uses_the_grasp_depth_for_both_pixels():
    """Both points must land at the same depth plane.

    If the approach pixel were back-projected at some other depth the
    resulting direction would tilt out of plane and the azimuth would be
    wrong by an amount that depends on the scene behind the object.
    """
    view, proj = _identity_camera()
    answer = ErGraspAnswer(
        grasp_uv=(320.0, 240.0), approach_uv=(420.0, 240.0), tilt_deg=60.0, roll_deg=0.0
    )
    pose = grasp_pose_from_answer(answer, 2.0, view, proj, WIDTH, HEIGHT)
    # Grasp pixel is the image centre, so it unprojects straight down the
    # camera's own -Z at the given depth.
    assert pose.xyz == pytest.approx((0.0, 0.0, -2.0))
    # approach pixel is +100 px in u at the same depth -> +0.4 m in camera x,
    # so travel runs from +x toward the object: bearing 180.
    assert abs(pose.azimuth_deg) == pytest.approx(180.0)
    assert pose.tilt_deg == pytest.approx(60.0)


def test_quaternion_matches_the_reported_angles():
    """`as_log`'s numbers and the commanded quaternion must not drift apart."""
    from task3_autonomy.er_grasp_orientation import approach_angles_from_quaternion

    view, proj = _identity_camera()
    answer = ErGraspAnswer(
        grasp_uv=(320.0, 240.0), approach_uv=(420.0, 300.0), tilt_deg=55.0, roll_deg=20.0
    )
    pose = grasp_pose_from_answer(answer, 1.5, view, proj, WIDTH, HEIGHT)
    tilt, azimuth, _ = approach_angles_from_quaternion(pose.quaternion_wxyz)
    assert tilt == pytest.approx(pose.tilt_deg, abs=1e-6)
    assert math.cos(math.radians(azimuth - pose.azimuth_deg)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "prim,expected",
    [
        ("plate2", "plate"),
        ("spoon2", "spoon"),
        ("simple_tray", "simple tray"),
        ("cup", "cup"),
        ("bowl2", "bowl"),
    ],
)
def test_natural_label_strips_scene_bookkeeping(prim, expected):
    """A vision model should be asked about a "plate", not a "plate2"."""
    from task3_autonomy.live_er_grasp import natural_label

    assert natural_label(prim) == expected


def test_natural_label_never_returns_empty():
    """An all-digit name would otherwise strip to nothing and ask ER-2 to
    find "" in the image."""
    from task3_autonomy.live_er_grasp import natural_label

    assert natural_label("2") == "2"


def test_long_axis_point_is_parsed_when_it_differs_from_the_grasp_point():
    answer = parse_grasp_answer(
        {
            "grasp_point": [500, 500],
            "approach_from": [500, 100],
            "long_axis_point": [500, 800],
            "tilt_deg": 20.0,
            "roll_deg": 0.0,
        },
        WIDTH,
        HEIGHT,
    )
    assert answer.long_axis_uv is not None


def test_long_axis_point_equal_to_the_grasp_point_is_ignored():
    """"No long axis" is expressed by repeating the grasp point; treating
    that as an axis would derive a roll from numerical noise."""
    answer = parse_grasp_answer(
        {
            "grasp_point": [500, 500],
            "approach_from": [500, 100],
            "long_axis_point": [500, 500],
            "tilt_deg": 20.0,
            "roll_deg": 15.0,
        },
        WIDTH,
        HEIGHT,
    )
    assert answer.long_axis_uv is None


def test_roll_across_long_axis_is_perpendicular_to_it():
    """The jaws must bite ACROSS an elongated object, not slide along it.

    ASPIRE's debugging table: "grasp succeeds but drops during lift ->
    gripper half-closed on slippery/elongated object -> try perpendicular
    yaw". Our spoon closes half-shut and drops, so this is the fix.
    """
    from task3_autonomy.er_grasp_orientation import quaternion_from_approach
    from task3_autonomy.live_er_grasp import roll_across_long_axis

    # Top-down grasp, object lying along world +X.
    roll = roll_across_long_axis((0, 0, 0), (0.1, 0.0, 0.0), 0.0, 0.0)
    assert roll is not None
    w, x, y, z = quaternion_from_approach(0.0, 0.0, roll)
    jaw = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    )
    # Jaw line must be near-perpendicular to the object's long axis (+X).
    assert abs(jaw[0]) < 0.2, jaw


def test_roll_across_long_axis_returns_none_for_a_degenerate_axis():
    from task3_autonomy.live_er_grasp import roll_across_long_axis

    assert roll_across_long_axis((0, 0, 0), (0, 0, 0), 0.0, 0.0) is None


def test_roll_across_long_axis_returns_none_when_axis_is_along_the_approach():
    """An object pointing straight at the gripper has no straddle direction
    the jaws can use, so the model's own roll should be kept."""
    from task3_autonomy.live_er_grasp import roll_across_long_axis

    assert roll_across_long_axis((0, 0, 0.1), (0, 0, 0), 0.0, 0.0) is None


def test_crop_box_stays_inside_the_image_and_keeps_its_size():
    from task3_autonomy.live_er_grasp import crop_box_around

    # Comfortably interior: centred exactly.
    box = crop_box_around(320.0, 240.0, 640, 480, 100)
    assert box == (220, 140, 420, 340)

    # Against every edge: shifted, never shrunk, never outside.
    for u, v in ((5.0, 5.0), (635.0, 475.0), (0.0, 240.0), (320.0, 479.0)):
        x0, y0, x1, y1 = crop_box_around(u, v, 640, 480, 100)
        assert 0 <= x0 < x1 <= 640
        assert 0 <= y0 < y1 <= 480
        assert (x1 - x0, y1 - y0) == (200, 200)


def test_crop_box_never_exceeds_a_small_image():
    from task3_autonomy.live_er_grasp import crop_box_around

    x0, y0, x1, y1 = crop_box_around(5.0, 5.0, 10, 8, 100)
    assert (x0, y0, x1, y1) == (0, 0, 10, 8)


def test_uv_from_crop_inverts_the_crop_offset():
    from task3_autonomy.live_er_grasp import crop_box_around, uv_from_crop

    box = crop_box_around(400.0, 300.0, 640, 480, 64)
    # The crop's own centre must map back to the point it was centred on.
    centre = ((box[2] - box[0]) / 2.0, (box[3] - box[1]) / 2.0)
    assert uv_from_crop(*centre, box) == (400.0, 300.0)


def test_answer_in_full_frame_moves_pixels_and_leaves_angles_alone():
    from task3_autonomy.live_er_grasp import (
        ErGraspAnswer,
        answer_in_full_frame,
    )

    box = (100, 50, 300, 250)
    answer = ErGraspAnswer(
        grasp_uv=(10.0, 20.0),
        approach_uv=(30.0, 40.0),
        tilt_deg=30.0,
        roll_deg=80.0,
        reason="stem",
        long_axis_uv=(50.0, 60.0),
    )
    moved = answer_in_full_frame(answer, box)

    assert moved.grasp_uv == (110.0, 70.0)
    assert moved.approach_uv == (130.0, 90.0)
    assert moved.long_axis_uv == (150.0, 110.0)
    assert moved.tilt_deg == 30.0
    assert moved.roll_deg == 80.0
    assert moved.reason == "stem"


def test_answer_in_full_frame_tolerates_a_missing_long_axis():
    from task3_autonomy.live_er_grasp import (
        ErGraspAnswer,
        answer_in_full_frame,
    )

    moved = answer_in_full_frame(
        ErGraspAnswer(
            grasp_uv=(1.0, 2.0),
            approach_uv=(3.0, 4.0),
            tilt_deg=0.0,
            roll_deg=0.0,
        ),
        (10, 20, 110, 120),
    )
    assert moved.grasp_uv == (11.0, 22.0)
    assert moved.long_axis_uv is None
