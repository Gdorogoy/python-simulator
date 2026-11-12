"""Acro-style plant: body-rate setpoints -> rate loop -> [thrust, torque] -> the shared mixer/integrator path.

Angle-style (the original drone): the action is [thrust, roll_torque, pitch_torque, yaw_torque] straight into
mixer_inversion. Any attitude hold lives in the PID outside the plant, as an angle error.
Acro-style: the action is [thrust, p_des, q_des, r_des] (body rates). The rate loop below runs every physics tick,
inside the plant, and never looks at an attitude angle, so nothing in the loop stops the drone from flipping.
See docs.md "Drone styles".
"""
import numpy as np

from app.dynamics.drone import QuadConfig, QuadState
from app.dynamics.angle_style_methods import mixer_inversion, timestamp_update

# Max commanded body rate per axis (p, q, r), rad/s. ~340 deg/s roll/pitch: a full flip in about 1 s.
MAX_BODY_RATE = (6.0, 6.0, 3.0)
RATE_ZETA = 0.7
# Fraction of the hover torque authority the rate loop may use; the rest is headroom for collective thrust.
TORQUE_MARGIN = 0.5


def rate_kp_from_plant(motor_tau: float, zeta: float = RATE_ZETA) -> float:
    """Pole placement for the rate loop, same idea as tune_pid.

    Plant per axis: I * w_dot = tau, with tau lagging the command by motor_tau. With tau_cmd = I * kp * (w_des - w)
    the closed loop is kp / (motor_tau s^2 + s + kp), so wn = 1 / (2 zeta motor_tau) and kp = wn^2 * motor_tau.
    """
    return 1.0 / (4.0 * zeta**2 * motor_tau)


def hover_torque_authority(config: QuadConfig) -> np.ndarray:
    """(3,) torque per axis when the whole hover thrust is shifted onto one side of the airframe."""
    hover_w = np.array(mixer_inversion(config, [config.mass * 9.81, 0.0, 0.0, 0.0]))
    f = np.array([r.k_f for r in config.rotors]) * hover_w**2
    m = np.array([r.k_m for r in config.rotors]) * hover_w**2
    x = np.array([r.position.x for r in config.rotors])
    y = np.array([r.position.y for r in config.rotors])
    return np.array([np.sum(np.abs(y) * f), np.sum(np.abs(x) * f), np.sum(m)])


class RateController:
    """Body-rate P(I) loop: tau = I * (kp * err + ki * integral), clipped to the feasible torque.

    gyro_ff adds w x (I w) so the loop cancels gyroscopic coupling. The numpy integrator does not model that term
    (PhysX does), so keep it off when flying the numpy plant.
    """

    def __init__(self, config: QuadConfig, kp=None, ki=0.0, integral_limit=1.0,
                 torque_margin=TORQUE_MARGIN, gyro_ff=False):
        self.inertia = np.array(config.inertia, dtype=np.float64)
        motor_tau = config.rotors[0].motor_tau
        self.kp = np.broadcast_to(np.asarray(kp if kp is not None else rate_kp_from_plant(motor_tau),
                                             dtype=np.float64), (3,)).copy()
        self.ki = np.broadcast_to(np.asarray(ki, dtype=np.float64), (3,)).copy()
        self.integral_limit = integral_limit
        self.torque_limit = torque_margin * hover_torque_authority(config)
        self.gyro_ff = gyro_ff
        self.integral = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)

    def compute_torque(self, ang_vel_body, rate_cmd, dt) -> np.ndarray:
        w = np.asarray(ang_vel_body, dtype=np.float64)
        err = np.asarray(rate_cmd, dtype=np.float64) - w
        self.integral = np.clip(self.integral + err * dt, -self.integral_limit, self.integral_limit)

        tau = self.inertia * (self.kp * err + self.ki * self.integral)
        if self.gyro_ff:
            tau = tau + np.cross(w, self.inertia * w)
        return np.clip(tau, -self.torque_limit, self.torque_limit)


def acro_timestamp_update(state: QuadState, config: QuadConfig, controller: RateController,
                          acro_action: list[float], wind_vector: list[float], dt: float) -> QuadState:
    """acro_action = [thrust (N, absolute), p_des, q_des, r_des]. One physics tick of the acro plant."""
    w = [state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]
    tau = controller.compute_torque(w, acro_action[1:4], dt)
    return timestamp_update(state, config, [acro_action[0], *tau], wind_vector, dt)
