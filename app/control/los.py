"""Line-of-sight velocity controller, batched torch, no Isaac imports. Shared by the LOS teacher (angle_rtl_train_isaac.make_los_label)
and the drone-threats in base_drone_env_isaac, so both fly with exactly the same maths.

    v_des      desired world velocity                      (caller decides: braking profile for the teacher, full speed for a chaser)
    a_cmd      = cruise_gain * (v_des - vel)               velocity loop, m/s^2
    thrust     = hover * (1 + a_cmd_z / g) / cos(tilt) - hover      (banked drone keeps its lift)
    roll_des, pitch_des = the (a_body / g) vector scaled to the tilt cone |roll, pitch| <= tan(max_tilt)   (a_body = a_cmd_xy rotated by -yaw; one isotropic clamp, not one per axis)
    torque     = kp_att * (att_des - att) - kd_att * rate  (roll, pitch);  kp_yaw * wrap(yaw_des - yaw) - kd_yaw * r  (yaw)
"""

from __future__ import annotations

import math

import torch

from app.control.torch_pid import wrap_angle

G = 9.81
CHASER_CRUISE_GAIN = 4.0     # velocity-loop gain of a chasing drone: the same as the teacher's (--teacher-cruise-gain 4), so it brakes and turns like the learner


def velocity_to_action(v_des, vel, ang_vel, roll, pitch, yaw, yaw_des, *, hover_thrust, kp_att, kd_att, kp_yaw, kd_yaw,
                       max_tilt, cruise_gain=1.0):
    """Angle-style action [thrust_delta, roll_torque, pitch_torque, yaw_torque] (N, 4) that steers `vel` toward `v_des`.
    Unclamped: the caller clamps to its action bounds. Returns (action, des_roll, des_pitch)."""
    accel_cmd = cruise_gain * (v_des - vel)
    cos_tilt = (torch.cos(roll) * torch.cos(pitch)).clamp_min(0.5)
    thrust_delta = hover_thrust * (1.0 + accel_cmd[:, 2] / G) / cos_tilt - hover_thrust

    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    accel_body_x = accel_cmd[:, 0] * cos_yaw + accel_cmd[:, 1] * sin_yaw
    accel_body_y = -accel_cmd[:, 0] * sin_yaw + accel_cmd[:, 1] * cos_yaw
    # Isotropic tilt clamp: the total tilt is atan(|a_xy| / g), not the larger of two axes. Clamping roll and pitch
    # separately let the first axis saturate and rotated the commanded direction (a diagonal target got a ~71 deg
    # tilt and undershot), so scale the (roll, pitch) vector to the tilt cone instead.
    raw_roll, raw_pitch = -accel_body_y / G, accel_body_x / G
    cap = math.tan(float(max_tilt))
    norm = torch.sqrt(raw_roll.pow(2) + raw_pitch.pow(2)).clamp_min(1e-9)
    scale = torch.clamp(cap / norm, max=1.0)
    des_roll = raw_roll * scale
    des_pitch = raw_pitch * scale

    roll_torque = kp_att * (des_roll - roll) - kd_att * ang_vel[:, 0]
    pitch_torque = kp_att * (des_pitch - pitch) - kd_att * ang_vel[:, 1]
    yaw_torque = kp_yaw * wrap_angle(yaw_des - yaw) - kd_yaw * ang_vel[:, 2]
    return torch.stack([thrust_delta, roll_torque, pitch_torque, yaw_torque], dim=-1), des_roll, des_pitch


def chase_velocity(rel, speed_cap):
    """Limit-LOS chaser: fly straight at the target at `speed_cap`. The speed limit is the ONLY limit here: no braking
    profile, no lead, no accel cap (the tilt clamp in velocity_to_action is the airframe's, not a chaser setting).
    rel = target - self (N,3), speed_cap (N,1) or scalar."""
    return rel / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6) * speed_cap


def hold_velocity(rel, max_speed=2.0, gain=1.0):
    """Position hold: P-control on the offset to a hold point, speed-limited. Used by a threat that is parked or still waiting."""
    v = gain * rel
    return v * torch.clamp(max_speed / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)
