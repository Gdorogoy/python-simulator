"""
mixer inversion, motor lag, thrust/torque integration.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from app.dynamics.drone import QuadConfig, Vector3D, QuadState, Quaternion

"""
Maps per-rotor speeds to [thrust, roll, pitch, yaw] via the mixer matrix M: row i holds
each rotor's contribution to output i (thrust, roll, pitch, yaw respectively).
"""
def mixer(config: QuadConfig, rotors_speed : list[float]):
    M=np.zeros((4,4))
    for i in range(0,4):
        for j in range(0,4):
            if i==0:
                M[i, j] = config.rotors[j].k_f
            if i==1:
                M[i, j] = config.rotors[j].position.y*config.rotors[j].k_f
            if i==2:
                M[i, j] = -(config.rotors[j].position.x) * config.rotors[j].k_f
            if i==3:
                M[i, j] =  config.rotors[j].k_m * config.rotors[j].spin_dir

    res=M @ rotors_speed

    return res


"""
Inverts the mixer matrix to go from a desired [thrust, roll, pitch, yaw] command
(e.g. from the RL policy) to the per-rotor speeds that produce it.
"""
def mixer_inversion(config: QuadConfig, desired_params : list[float]) -> list[float]:
    M = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            if i == 0:
                M[i, j] = config.rotors[j].k_f
            if i == 1:
                M[i, j] = config.rotors[j].position.y * config.rotors[j].k_f
            if i == 2:
                M[i, j] = -(config.rotors[j].position.x) * config.rotors[j].k_f
            if i == 3:
                M[i, j] = config.rotors[j].k_m * config.rotors[j].spin_dir

    speeds=np.linalg.inv(M) @ desired_params
    speeds = np.clip(speeds, 0.0, None)
    res = [np.sqrt(speed) for speed in speeds]

    return res

"""
First-order lag toward the target rpm (motor_tau: small = responsive, large = sluggish),
so a rotor can't jump speed instantly, like a real motor's spin-up/down time.
"""
def motor_lag(w_current: float, w_target: float, motor_tau: float, dt: float) -> float:
    w_dot= (w_target-w_current) / motor_tau
    w_new = w_current + w_dot * dt
    return w_new


"""
Thrust from one rotor, along body z. Always positive since it's squared in omega, so
spin direction (CW/CCW) doesn't matter here the way it does for torque.
"""

def thrust(k_f: float, w_i: float):
    F_i=k_f*w_i**2
    return F_i

"""
Reaction torque from one rotor about the vertical axis. Unlike thrust, torque direction
depends on spin_dir (+1/-1 for CW/CCW), since drag reaction opposes the spin.
"""

def torque(k_m: float, w_i: float, spin_dir_i: float):
    F_i= k_m*w_i**2 * spin_dir_i
    return F_i


"""Sums each rotor's thrust into the net body thrust."""

def net_combining_thrust(config: QuadConfig, rotor_speeds: list[float]):
    F_net=np.zeros(4)
    for i in range(0,4):
        F_net[i]=thrust(config.rotors[i].k_f, rotor_speeds[i])

    return sum(F_net)
"""Sums each rotor's moment-arm contribution and reaction torque into net body torque."""
def net_combining_torque(config: QuadConfig, rotor_speeds: list[float]):
    F_net = np.zeros(3)

    for i in range(0, 4):
        rotor = config.rotors[i]
        F_i = thrust(rotor.k_f, rotor_speeds[i])
        r_i = np.array([rotor.position.x, rotor.position.y, rotor.position.z])
        F_vec = np.array([0, 0, F_i])
        F_net += np.cross(r_i, F_vec)  # moment-arm contribution
        F_net += np.array([0, 0, torque(rotor.k_m, rotor_speeds[i], rotor.spin_dir)])  # reaction torque
    return F_net

"""Angular acceleration per axis: torque divided by that axis's inertia."""
def angular_acceleration(config: QuadConfig, net_torque: list[float]) -> np.ndarray:
    alpha = np.zeros(3)
    for i in range(3):
        alpha[i] = net_torque[i] / config.inertia[i]
    return alpha


"""Integrates angular velocity forward by alpha * dt."""
def update_angular_velocity(state: QuadState, alpha: np.ndarray, dt: float) -> np.ndarray:
    current_w = np.array([state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z])
    new_w = current_w + alpha * dt
    return new_w



"""Drag force opposing the drone's current velocity vector."""
def drag_force(velocity: list[float] , drag_coeff : float , cross_sec_area : float , air_dens : float):
    v=np.array(velocity)
    speed=np.linalg.norm(v)

    if speed<0.01:
        return np.zeros(3)

    F_drag= 0.5* air_dens * speed**2 * drag_coeff * cross_sec_area

    return (-v/speed) *  F_drag


"""Per-axis wind force from the wind vector, scaled by mass and a wind coefficient."""
def wind(wind: list[float] , mass: float , k_wind_coeff : float):
    F_wind = np.zeros(3)
    for i in range(0,3):
        F_wind[i]=wind[i] * mass * k_wind_coeff

    return F_wind




def timestamp_update(state: QuadState, config: QuadConfig , rl_action: list[float] , wind_vector : list[float] ,dt: float):

    w_target : list[float] = mixer_inversion(config, rl_action)

    w_actual = np.zeros(4)
    for i in range(0,4):
        w_actual[i] = motor_lag(state.rotor_rpm[i], w_target[i], config.rotors[i].motor_tau, dt)

    net_thrust = net_combining_thrust(config, w_actual)  # scalar
    net_torque = net_combining_torque(config, w_actual)  # 3-element vector

    drag = drag_force(
        velocity=[state.velocity.x, state.velocity.y, state.velocity.z],
        drag_coeff=config.drag_coeff,
        cross_sec_area=0.05,  # TODO: promote to a proper QuadConfig field
        air_dens=1.225,
    )

    wind_force = wind(
        wind=wind_vector,
        mass=config.mass,
        k_wind_coeff=0.1,  # tunable constant
    )

    gravity_force = np.array([0, 0, -config.mass * 9.81])

    current_rot = Rotation.from_quat(
        [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w])

    thrust_body_frame = np.array([0, 0, net_thrust])

    thrust_world_frame = current_rot.apply(thrust_body_frame)

    total_force = thrust_world_frame + drag + wind_force + gravity_force

    linear_accel = total_force / config.mass

    new_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z]) + linear_accel * dt

    new_position = np.array([state.position.x, state.position.y, state.position.z]) + new_velocity * dt

    alpha = angular_acceleration(config, net_torque)

    new_angular_velocity = update_angular_velocity(state, alpha, dt)

    delta_rot = Rotation.from_rotvec(new_angular_velocity * dt)

    new_rot = delta_rot * current_rot

    new_quat = new_rot.as_quat()  # returns [x, y, z, w]

    quad_state=QuadState(
        position=Vector3D(*new_position),
        velocity=Vector3D(*new_velocity),
        orientation=Quaternion(*new_quat),
        angular_velocity=Vector3D(*new_angular_velocity),
        rotor_rpm=list(w_actual),
    )

    return quad_state
