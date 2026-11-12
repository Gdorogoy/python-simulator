import math

import torch

from app.reward_functions.angle_reward_rc import RewardRCConfig, compute_reward_rc

CFG = RewardRCConfig()


def run(dist, prev, stall=0, zone=0, first=False, timed_out=False, tilt_deg=0.0, roll_deg=0.0, pitch_deg=0.0, icpt=None, pos=(0.0, 0.0, 10.0),
        cfg=CFG, hit_done=False, post=0):
    f = lambda v: torch.tensor([float(v)])
    rel = None if icpt is None else torch.tensor([icpt], dtype=torch.float32)
    return compute_reward_rc(pos=torch.tensor([pos]), dist=f(dist), prev_dist=f(prev), start_dist=f(100.0),
                             tilt=f(math.radians(tilt_deg)), roll=f(math.radians(roll_deg)),
                             pitch=f(math.radians(pitch_deg)), interceptor_rel=rel,
                             timed_out=torch.tensor([timed_out]), stall_count=torch.tensor([stall]),
                             zone_count=torch.tensor([zone]), first_entry=torch.tensor([first]),
                             grace_steps=60, limit_steps=240, limit_steps_hit=120, cfg=cfg,
                             hit_done=torch.tensor([hit_done]), post_hit_count=torch.tensor([post]))


def test_closing_and_retreat():
    up = run(dist=49.0, prev=50.0).reward.item()
    down = run(dist=51.0, prev=50.0).reward.item()
    assert up > -CFG.c_step > down
    assert not run(dist=51.0, prev=50.0).terminated.item()


def test_potential_telescopes():
    a = run(dist=100.0, prev=200.0).parts["closing"] + run(dist=50.0, prev=100.0).parts["closing"]
    b = run(dist=50.0, prev=200.0).parts["closing"]
    assert abs(a.item() - b.item()) < 1e-5


def test_hover_and_timeout():
    assert abs(run(50, 50, stall=61).parts["hover"].item() + CFG.c_hover) < 1e-6
    out = run(50, 50, stall=240)
    assert out.hover_term.item() and out.terminated.item()
    assert abs(out.parts["hover"].item() - (-CFG.c_hover - CFG.r_hover)) < 1e-6
    assert abs(run(50, 50, timed_out=True).parts["timeout"].item() + CFG.r_timeout) < 1e-6


def test_zone_entry_then_hold_hit():
    entry = run(dist=4.9, prev=5.2, zone=1, first=True)
    assert not entry.hit.item() and not entry.terminated.item()
    assert abs(entry.parts['zone_entry'].item() - CFG.r_zone_entry) < 1e-6
    holding = run(dist=4.9, prev=4.9, zone=60)
    assert not holding.hit.item() and holding.parts['zone_entry'].item() == 0.0
    old = RewardRCConfig(post_hit_steps=0)                      # 0 = end on the hit
    done = run(dist=4.9, prev=4.9, zone=120, cfg=old)
    assert done.hit.item() and done.terminated.item()
    assert abs(done.parts['hit'].item() - (CFG.r_hit - CFG.r_zone_entry)) < 1e-6
    assert CFG.r_hit == 45.0 and CFG.post_hit_steps == 30


def test_post_hit_window():
    scored = run(dist=4.9, prev=4.9, zone=120)                  # default: the hit pays once and the flight goes on
    assert scored.hit.item() and not scored.terminated.item()
    assert abs(scored.parts['hit'].item() - (CFG.r_hit - CFG.r_zone_entry)) < 1e-6
    # inside the window: flying away from the target (closing / axis would punish it), slow, timed out, stalled -> nothing is paid
    inside = run(dist=9.0, prev=4.9, zone=0, hit_done=True, post=5, stall=500, timed_out=True)
    assert not inside.hit.item() and not inside.terminated.item() and not inside.hover_term.item()
    assert inside.reward.item() == 0.0
    over = run(dist=40.0, prev=30.0, hit_done=True, post=CFG.post_hit_steps)
    assert over.terminated.item() and over.reward.item() == 0.0 and not over.hit.item()
    assert not run(dist=40.0, prev=30.0, hit_done=True, post=CFG.post_hit_steps - 1).terminated.item()
    # crash, out of bounds and being caught still end it with their penalty (plus the usual step cost, as outside the window)
    crash = run(dist=9.0, prev=4.9, hit_done=True, post=5, tilt_deg=61)
    assert crash.terminated.item() and abs(crash.reward.item() + CFG.r_crash + CFG.c_step) < 1e-6
    oob = run(dist=9.0, prev=4.9, hit_done=True, post=5, pos=(0.0, 0.0, -1.0))
    assert oob.terminated.item() and oob.oob.item()
    caught = run(dist=9.0, prev=4.9, hit_done=True, post=5, icpt=(0.1, 0.1, 0.1))
    assert caught.intercepted.item() and abs(caught.reward.item() + CFG.r_intercepted + CFG.c_step) < 1e-6
    # the whole hit pays 45: 7 on entry + 38 on completion
    assert CFG.r_zone_entry + (CFG.r_hit - CFG.r_zone_entry) == 45.0


def test_intercepted_beats_hit():
    out = run(dist=4.0, prev=4.2, icpt=(0.1, 0.1, 0.1))
    assert out.intercepted.item() and not out.hit.item() and out.terminated.item()
    assert abs(out.reward.item() - (-CFG.r_intercepted - CFG.c_step)) < 1e-6
    assert not run(dist=4.0, prev=4.2, icpt=(0.3, 0.0, 0.0)).intercepted.item()


def test_attitude_limits():
    assert run(50, 50, tilt_deg=61).tilt_bad.item()
    assert not run(50, 50, tilt_deg=59).terminated.item()
    assert run(50, 50, roll_deg=86).roll_bad.item()
    assert run(50, 50, pitch_deg=121).pitch_bad.item()
    assert run(50, 50, pos=(0.0, 0.0, -1.0)).oob.item()


def test_zone_timer_features():
    from app.reward_functions.angle_reward_rc import zone_timer_features
    dist = torch.tensor([20.0, 4.9, 4.9, 4.9, 5.1])
    count = torch.tensor([0, 1, 60, 500, 0])
    f = zone_timer_features(dist, count, hit_radius=5.0, limit_steps_hit=120)
    assert f.shape == (5, 2)
    assert f[:, 0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert torch.allclose(f[:, 1], torch.tensor([0.0, 1 / 120, 0.5, 1.0, 0.0]))


def test_axis_progress():
    cfg = RewardRCConfig(axis_k=1.0)

    def axis(rel, prev_rel):
        t = lambda v: torch.tensor([v], dtype=torch.float32)
        out = compute_reward_rc(pos=torch.tensor([[0.0, 0.0, 10.0]]), dist=t(1.0), prev_dist=t(1.0), start_dist=t(100.0),
                                tilt=t(0.0), roll=t(0.0), pitch=t(0.0), interceptor_rel=None, timed_out=torch.tensor([False]),
                                stall_count=torch.tensor([0]), zone_count=torch.tensor([0]), first_entry=torch.tensor([False]),
                                grace_steps=60, limit_steps=240, limit_steps_hit=120, cfg=cfg, rel=torch.tensor([rel]),
                                prev_rel=torch.tensor([prev_rel]))
        return out.parts["axis"].item()

    assert axis((4.0, 5.0, 6.0), (4.0, 5.0, 6.0)) == 0.0
    assert axis((3.0, 5.0, 6.0), (4.0, 5.0, 6.0)) > 0.0                       # closed on x only
    assert axis((4.0, 5.0, 6.0), (3.0, 5.0, 6.0)) < 0.0                       # backed off on x
    out_and_back = axis((6.0, 5.0, 6.0), (4.0, 5.0, 6.0)) + axis((4.0, 5.0, 6.0), (6.0, 5.0, 6.0))
    assert abs(out_and_back) < 1e-6                                           # telescopes: dithering earns nothing
    assert axis((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)) > 0.0                       # z error shrinking to 0 pays
    assert RewardRCConfig().axis_k == 0.5                                     # on by default


def test_los_term():
    cfg = RewardRCConfig(los_k=1.0, los_mode="cos")

    def los(vel, rel=(10.0, 0.0, 0.0), k_cfg=cfg):
        t = lambda v: torch.tensor([v], dtype=torch.float32)
        out = compute_reward_rc(pos=torch.tensor([[0.0, 0.0, 10.0]]), dist=t(10.0), prev_dist=t(10.0), start_dist=t(100.0),
                                tilt=t(0.0), roll=t(0.0), pitch=t(0.0), interceptor_rel=None, timed_out=torch.tensor([False]),
                                stall_count=torch.tensor([0]), zone_count=torch.tensor([0]), first_entry=torch.tensor([False]),
                                grace_steps=60, limit_steps=240, limit_steps_hit=120, cfg=k_cfg, rel=torch.tensor([rel]),
                                prev_rel=torch.tensor([rel]), vel=torch.tensor([vel]))
        return out.parts["los"].item()

    assert abs(los((3.0, 0.0, 0.0)) - 1.0) < 1e-4                  # straight at the target
    assert abs(los((-3.0, 0.0, 0.0)) + 1.0) < 1e-4                 # straight away
    assert abs(los((0.0, 3.0, 0.0))) < 1e-4                        # sideways
    assert los((0.1, 0.0, 0.0)) == 0.0                             # below los_min_speed: no direction signal
    assert los((3.0, 0.0, 0.0), k_cfg=RewardRCConfig(los_k=0.0)) == 0.0
    assert RewardRCConfig().los_min_speed == 0.5


def _smoothstep_ref(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def los_penalty(deg, speed=5.0, gate=None, cfg=None):
    cfg = cfg or RewardRCConfig(los_k=0.02)
    t = lambda v: torch.tensor([v], dtype=torch.float32)
    a = math.radians(deg)
    vel = torch.tensor([[speed * math.cos(a), speed * math.sin(a), 0.0]])
    out = compute_reward_rc(pos=torch.tensor([[0.0, 0.0, 10.0]]), dist=t(10.0), prev_dist=t(10.0), start_dist=t(100.0),
                            tilt=t(0.0), roll=t(0.0), pitch=t(0.0), interceptor_rel=None, timed_out=torch.tensor([False]),
                            stall_count=torch.tensor([0]), zone_count=torch.tensor([0]), first_entry=torch.tensor([False]),
                            grace_steps=60, limit_steps=240, limit_steps_hit=120, cfg=cfg,
                            rel=torch.tensor([[10.0, 0.0, 0.0]]), prev_rel=torch.tensor([[10.0, 0.0, 0.0]]), vel=vel,
                            los_gate=None if gate is None else torch.tensor([gate]))
    return out.parts["los"].item()


def test_los_penalty_bands():
    assert abs(los_penalty(0) + 0.02) < 1e-6                                      # aligned: m = 1
    assert abs(los_penalty(35) + 0.02) < 1e-6                                     # still m = 1 at the first edge
    assert abs(los_penalty(60) + 0.02 * (1 + _smoothstep_ref(35, 90, 60))) < 1e-6
    assert abs(los_penalty(90) + 0.04) < 1e-6                                     # sideways: m = 2
    assert abs(los_penalty(135) + 0.10) < 1e-6                                    # turning back: m = 5
    assert abs(los_penalty(180) + 0.10) < 1e-6
    assert los_penalty(0, speed=0.1) == 0.0                                       # too slow to have a direction
    assert los_penalty(0, gate=False) == 0.0                                      # not handed off / not in the far half
    assert abs(los_penalty(0, cfg=RewardRCConfig(los_k=0.02, los_mode="cos")) - 0.02) < 1e-6   # old cos mode: +k when aligned
    off = run(dist=9.0, prev=4.9, hit_done=True, post=5, pos=(0.0, 0.0, 10.0))  # inside the post-hit window: nothing
    assert off.parts["los"].item() == 0.0
