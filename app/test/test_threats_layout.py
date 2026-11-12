import numpy as np

from app.environmental.threats import Threats, ThreatDroneEnv, layout_from_obs_dim
from app.reward_functions.rewards import reward_func


def test_layout_from_obs_dim():
    assert layout_from_obs_dim(23) == (0, False)
    assert layout_from_obs_dim(58) == (5, False)
    assert layout_from_obs_dim(25) == (0, True)
    assert layout_from_obs_dim(60) == (5, True)
    for bad in (22, 24, 26, 61):
        try:
            layout_from_obs_dim(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_sphere_spawn_shell_and_speed_range():
    t = Threats(4, 4, max_speed=12.0, min_speed=6.0, spread=30.0, spawn_min=15.0, rng=np.random.default_rng(0))
    base = np.array([5.0, -3.0, 4.0])
    radii, caps = [], []
    for _ in range(500):
        t.reset(base, [0.0, 0.0, 5.0])
        radii += list(np.linalg.norm(t.pos - base, axis=1))
        caps += list(t.speed_cap)
        assert (t.pos[:, 2] >= 0).all()
    r = np.array(radii)
    assert r.max() <= 30.0 + 1e-6
    assert (r >= 15.0 - 1e-6).mean() > 0.9                           # only underground-mirrored points can land closer
    assert len(set(np.round(t.pos[:, 0], 3))) > 1                    # not one place
    assert 6.0 <= min(caps) and max(caps) <= 12.0 and max(caps) - min(caps) > 3.0


def test_env_obs_size_with_timer_and_release_wait():
    env = ThreatDroneEnv(reward_func, pid_gains_path=None, max_steps=200)
    env.set_timer(1.0, 2.0)
    env.set_threats(Threats(5, 1, rng=np.random.default_rng(1)))
    obs, _ = env.reset()
    assert obs.shape == (23 + 7 * 5 + 2,) == env.observation_space.shape
    assert obs[-2:].tolist() == [0.0, 0.0] or obs[-2] in (0.0, 1.0)
    before = env.threats.pos.copy()
    obs, *_ = env.step(np.zeros(4, dtype=np.float32))
    assert obs.shape == (60,)
    assert np.allclose(env.threats.pos, before)                      # threats_moving is False: nobody moves


def test_base_spawn_clearance_and_ring():
    for slots, radius in ((2, None), (3, None), (5, None), (5, 1.0), (7, 1.0)):
        t = Threats(slots, slots, spawn="base", radius=radius, buffer=0.5, rng=np.random.default_rng(3))
        gap = 2 * t.body_radius + 0.5
        for _ in range(20):
            t.reset([4.0, -2.0, 6.0], [0.0, 0.0, 5.0])
            assert np.allclose(t.pos[0], [4.0, -2.0, 3.0])                 # slot 0 sits on the base at spawn_z
            assert np.allclose(t.pos[:, 2], 3.0)                           # one height: the ring is horizontal
            d = np.linalg.norm(t.pos[:, None] - t.pos[None], axis=-1) + np.eye(slots) * 1e9
            assert d.min() >= gap - 1e-9, (slots, radius, d.min(), gap)


def test_release_range_and_latch():
    t = Threats(4, 4, release_frac_range=(0.01, 0.67), rng=np.random.default_rng(5))
    fracs = []
    for _ in range(200):
        t.reset([0.0, 0.0, 5.0], [0.0, 0.0, 5.0])
        fracs += list(t.release_frac_slot)
    assert 0.01 <= min(fracs) and max(fracs) <= 0.67 and max(fracs) - min(fracs) > 0.4
    t.reset([0.0, 0.0, 5.0], [0.0, 0.0, 5.0])
    t.release_frac_slot = np.array([0.1, 0.3, 0.5, 0.7])
    t.update_release(left=60.0, ref=100.0)                                 # 60% left: only the 0.7 one
    assert t.released.tolist() == [False, False, False, True]
    t.update_release(left=40.0, ref=100.0)
    assert t.released.tolist() == [False, False, True, True]
    t.update_release(left=90.0, ref=100.0)                                 # latched: moving away does not un-release
    assert t.released.tolist() == [False, False, True, True]


def test_chaser_is_a_drone_that_waits_then_closes_in():
    t = Threats(2, 2, max_speed=8.0, spread=30.0, spawn_min=20.0, rng=np.random.default_rng(7))
    drone = np.array([0.0, 0.0, 10.0])
    t.reset(drone, drone)
    start = t.pos.copy()
    for _ in range(240):
        t.step(drone, np.zeros(3), 1 / 240)
    assert np.abs(t.pos - start).max() < 0.2                               # not released: hovers where it spawned
    t.released[:] = True
    d0 = np.linalg.norm(t.pos - drone, axis=1)
    peak = 0.0
    for _ in range(240 * 6):
        t.step(drone, np.zeros(3), 1 / 240)
        peak = max(peak, np.linalg.norm(t.vel, axis=1).max())
    d1 = np.linalg.norm(t.pos - drone, axis=1)
    assert (d1 < d0 - 5.0).all()                                           # closed in
    assert peak > 3.0 and peak < 8.0 * 1.5                                 # got up to speed, near the commanded cap, no hard clamp


def test_released_chaser_catches_an_idle_drone_through_the_env():
    env = ThreatDroneEnv(reward_func, pid_gains_path=None, max_steps=4000)
    env.set_threats(Threats(3, 1, max_speed=10.0, spawn_min=15.0, spread=25.0, rng=np.random.default_rng(11)))
    obs, _ = env.reset(start_pos=np.array([0.0, 0.0, 10.0]), target_pos=np.array([0.0, 0.0, 30.0]))   # far from the target: no Hit
    assert obs.shape == (23 + 7 * 3,)
    env.threats.released[:] = True
    reason = None
    for _ in range(4000):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        if terminated or truncated:
            reason = info["reason"]
            break
    assert reason == "Intercepted", reason
    assert reward == ThreatDroneEnv.R_INTERCEPTED
