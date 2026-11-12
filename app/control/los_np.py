"""Numpy mirror of app/control/los.py for the web viewer sim (no torch). Same maths line for line; app/test/test_los_np.py diffs the two.
All arrays are batched over the first axis (one row per drone)."""

from __future__ import annotations

import math

import numpy as np

G = 9.81
CHASER_CRUISE_GAIN = 4.0     # velocity-loop gain of a chasing drone: the same as the teacher's (--teacher-cruise-gain 4), so it brakes and turns like the learner


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def velocity_to_action(v_des, vel, ang_vel, roll, pitch, yaw, yaw_des, *, hover_thrust, kp_att, kd_att, kp_yaw, kd_yaw,
                       max_tilt, cruise_gain=1.0):
    """Angle-style action [thrust_delta, roll_torque, pitch_torque, yaw_torque] (N, 4) steering `vel` toward `v_des`; unclamped.
    Returns (action, des_roll, des_pitch)."""
    accel_cmd = cruise_gain * (v_des - vel)
    cos_tilt = np.maximum(np.cos(roll) * np.cos(pitch), 0.5)
    thrust_delta = hover_thrust * (1.0 + accel_cmd[:, 2] / G) / cos_tilt - hover_thrust

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    accel_body_x = accel_cmd[:, 0] * cos_yaw + accel_cmd[:, 1] * sin_yaw
    accel_body_y = -accel_cmd[:, 0] * sin_yaw + accel_cmd[:, 1] * cos_yaw
    # isotropic tilt clamp: scale the (roll, pitch) vector to the cone |a_xy| / g <= tan(max_tilt)
    raw_roll, raw_pitch = -accel_body_y / G, accel_body_x / G
    cap = math.tan(float(max_tilt))
    norm = np.maximum(np.sqrt(raw_roll ** 2 + raw_pitch ** 2), 1e-9)
    scale = np.minimum(cap / norm, 1.0)
    des_roll = raw_roll * scale
    des_pitch = raw_pitch * scale

    roll_torque = kp_att * (des_roll - roll) - kd_att * ang_vel[:, 0]
    pitch_torque = kp_att * (des_pitch - pitch) - kd_att * ang_vel[:, 1]
    yaw_torque = kp_yaw * wrap_angle(yaw_des - yaw) - kd_yaw * ang_vel[:, 2]
    return np.stack([thrust_delta, roll_torque, pitch_torque, yaw_torque], axis=-1), des_roll, des_pitch


def chase_velocity(rel, speed_cap):
    """Limit-LOS chaser: straight at the target at `speed_cap`; the speed is the only limit. rel (N,3), speed_cap (N,1) or scalar."""
    return rel / np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-6) * speed_cap


def hold_velocity(rel, max_speed=2.0, gain=1.0):
    """Position hold: P-control on the offset to a hold point, speed-limited."""
    v = gain * rel
    return v * np.minimum(max_speed / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6), 1.0)
