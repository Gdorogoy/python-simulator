from types import SimpleNamespace

from app.control.two_phase import CarrotController, load_gains_by_dist


def vec(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def state(pos, vel=(0.0, 0.0, 0.0)):
    return SimpleNamespace(position=vec(*pos), velocity=vec(*vel), angular_velocity=vec(0.0, 0.0, 0.0),
                           orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))


def controller(target):
    carrot = CarrotController(load_gains_by_dist(), cruise_speed=18.0, handoff_speed=9.0)
    carrot.reset(state((0.0, 0.0, 5.0)), target, switch_dist=10.0)
    return carrot


def test_on_the_line_still_flies_straight_along_it():
    target = (100.0, 0.0, 5.0)
    action = controller(target).compute_action(state((50.0, 0.0, 5.0), (5.0, 0.0, 0.0)), target, 0.0)
    assert abs(action[1]) < 1e-9                    # no roll: nothing to correct sideways
    assert action[2] > 0.0                          # pitch forward to speed up toward 100 m


def test_off_the_line_steers_back_toward_the_target():
    # 30 m off the line to +y, level with the target in x: the target is to the -y side, so the drone must roll that way.
    # The old fixed-direction controller commanded 0 roll here and carried the sideways error to the end.
    target = (100.0, 0.0, 5.0)
    action = controller(target).compute_action(state((50.0, 30.0, 5.0)), target, 0.0)
    assert action[1] > 0.0

