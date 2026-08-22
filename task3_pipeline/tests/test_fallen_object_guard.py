"""The fallen-object predicate: parameter-free, so it is worth pinning."""
from types import SimpleNamespace

from task3_pipeline.world_isaac import IsaacWorld


def _world(spawn, current):
    w = object.__new__(IsaacWorld)
    w._spawn_object_z = dict(spawn)
    w.object_position = lambda name: (0.0, 0.0, current[name])
    return w


def test_object_still_on_its_counter_is_not_fallen():
    w = _world({"cup": 0.747}, {"cup": 0.7465})
    assert w._object_has_fallen("cup") is False


def test_object_on_the_floor_is_fallen():
    w = _world({"cup": 0.747}, {"cup": 0.0352})
    assert w._object_has_fallen("cup") is True


def test_object_lifted_above_its_spawn_is_not_fallen():
    """A grasped, lifted object must never read as fallen."""
    w = _world({"cup": 0.747}, {"cup": 0.95})
    assert w._object_has_fallen("cup") is False


def test_exactly_halfway_counts_as_still_up():
    """Ties go to 'not fallen' -- the guard should never skip an object it
    is not sure about."""
    w = _world({"cup": 0.8}, {"cup": 0.4})
    assert w._object_has_fallen("cup") is False


def test_unknown_object_is_never_skipped():
    w = _world({}, {"cup": 0.0})
    assert w._object_has_fallen("cup") is False


def test_reach_refuses_a_fallen_object_before_driving():
    """The guard has to sit in reach(), not only in the live-ER path.

    Returning None from _live_er_grasp_pose means "use the fallback grasp
    target", not "skip this object" -- run 20 logged the fallen cup and then
    reached and closed on it anyway, ending 0.918 m away from an object on
    the floor.
    """
    w = _world({"cup": 0.7471}, {"cup": 0.0353})
    w._m = {}
    w._log_phase = lambda *a, **k: None
    w._active_object = None

    result = IsaacWorld.reach(w, "left", "cup")

    assert result["reason"] == "object_has_fallen_off_its_surface"
    assert result["strict_reach"] is False
    assert result["position_error_m"] == 999.0


def test_reach_does_not_refuse_an_object_still_on_its_surface():
    """A standing object must fall through to the real reach path.

    It will raise on the mock's missing internals -- that is the point: it
    got PAST the guard.
    """
    import pytest

    w = _world({"cup": 0.7471}, {"cup": 0.7465})
    w._m = {}
    w._log_phase = lambda *a, **k: None
    w._active_object = None

    with pytest.raises(Exception):
        IsaacWorld.reach(w, "left", "cup")


def test_grasp_refuses_a_fallen_object_too():
    """reach and grasp are separate skills -- refusing one does not refuse
    the other. Run 21 skipped the reach for the fallen cup and then closed
    the gripper anyway with the object 2.766 m away.
    """
    w = _world({"cup": 0.7471}, {"cup": 0.0339})
    w._m = {}
    w._log_phase = lambda *a, **k: None
    w._active_object = None

    result = IsaacWorld.grasp(w, "left", "cup")

    assert result["held"] is False
    assert result["scored"] is False
    assert result["reason"] == "object_has_fallen_off_its_surface"
