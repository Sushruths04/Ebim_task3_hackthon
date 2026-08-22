import math
from task3_autonomy.skills import RotateTo
from task3_autonomy.navigation import Pose2D


def test_min_creep_floors_a_decayed_rate():
    """The measured stall: 4.13 deg of error, kp*error = 0.108 rad/s."""
    skill = RotateTo(math.radians(4.13), yaw_tolerance_rad=math.radians(4.0),
                     min_creep_radps=0.08)
    wz, done = skill.compute(Pose2D(0.0, 0.0, 0.0))
    assert not done
    assert abs(wz) >= 0.08


def test_default_is_byte_identical_to_the_old_p_controller():
    for err_deg in (10.0, -30.0, 90.0, -150.0):
        skill = RotateTo(math.radians(err_deg))
        wz, done = skill.compute(Pose2D(0.0, 0.0, 0.0))
        expected = max(-0.5, min(0.5, 1.5 * math.radians(err_deg)))
        assert not done
        assert wz == expected, (err_deg, wz, expected)


def test_creep_never_exceeds_max_yaw_rate():
    skill = RotateTo(math.pi, min_creep_radps=99.0, max_yaw_rate=0.5)
    wz, _ = skill.compute(Pose2D(0.0, 0.0, 0.0))
    assert abs(wz) <= 0.5


def test_creep_keeps_the_sign_of_the_error():
    skill = RotateTo(math.radians(-4.13), yaw_tolerance_rad=math.radians(4.0),
                     min_creep_radps=0.08)
    wz, _ = skill.compute(Pose2D(0.0, 0.0, 0.0))
    assert wz < 0


def test_inside_tolerance_still_reports_done_and_zero():
    skill = RotateTo(math.radians(1.0), yaw_tolerance_rad=math.radians(4.0),
                     min_creep_radps=0.08)
    assert skill.compute(Pose2D(0.0, 0.0, 0.0)) == (0.0, True)
