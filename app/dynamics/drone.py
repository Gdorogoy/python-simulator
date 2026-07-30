"""
Drone dynamics entities.

Three core dataclasses:
    RotorConfig — static physics of a single motor/rotor
    QuadConfig  — static physics of the whole drone (mass, inertia, 4 rotors)
    QuadState   — dynamic per-tick state (position, velocity, orientation, etc)

No redundant classes. CreateDrone/Rotor were removed — QuadConfig IS the
"create drone" representation, built via create_quad_config() below.
"""

import math
from dataclasses import dataclass


@dataclass
class Vector3D:
    x: float
    y: float
    z: float


@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

"""Static, per-motor. One of these per rotor (4 total for a quadcopter)."""
@dataclass
class RotorConfig:
    position: Vector3D    # relative to center of mass
    spin_dir: int          # +1 = clockwise, -1 = counter-clockwise
    k_f: float              # thrust coefficient: F = k_f * omega^2
    k_m: float               # torque coefficient: M = k_m * omega^2
    max_rpm: float
    motor_tau: float         # spin-up/down time constant (seconds)

"""Static, per drone type. Describes the whole airframe."""
@dataclass
class QuadConfig:
    mass: float
    inertia: tuple[float, float, float]  # Ixx, Iyy, Izz
    arm_length: float
    drag_coeff: float
    rotors: list[RotorConfig]  # always 4, in X configuration

"""Dynamic, changes every simulation tick."""
@dataclass
class QuadState:
    position: Vector3D
    velocity: Vector3D
    orientation: Quaternion
    angular_velocity: Vector3D   # p, q, r in body frame
    rotor_rpm: list[float]        # current actual rpm, 4 values


"""
Places 4 rotors in a standard X configuration, 90 degrees apart,
with alternating spin direction (CW/CCW) so reaction torques
cancel out during hover — this is real quadcopter physics, not
optional cosmetics.
"""

def create_quad_rotors(
    arm_length: float,
    max_rpm: float,
    motor_tau: float,
    mass: float,
    kf_km_ratio: float =0.02,
    hover_rpm_fraction: float = 0.5,
) -> list[RotorConfig]:

    angles_deg = [45, 135, 225, 315]
    spin_dirs = [1, -1, 1, -1]  # alternating — cancels net yaw torque

    # finiding k_f so 4 rotors will balance the gravity
    k_f=(mass*9.81/4)/ (max_rpm*hover_rpm_fraction)**2

    rotors: list[RotorConfig] = []

    for angle_deg, spin_dir in zip(angles_deg, spin_dirs):
        angle_rad = math.radians(angle_deg)
        rotor_position = Vector3D(
            x=arm_length * math.cos(angle_rad),
            y=arm_length * math.sin(angle_rad),
            z=0.0,
        )
        rotors.append(RotorConfig(
            position=rotor_position,
            spin_dir=spin_dir,
            k_f=k_f ,
            k_m=k_f*kf_km_ratio, # yaw torque coefficient, scaled off k_f
            max_rpm=max_rpm,
            motor_tau=motor_tau,
        ))

    return rotors



"""
Builds a complete QuadConfig from raw physical parameters.
This replaces the old CreateDrone class — no intermediate
redundant object, just parameters straight into the final config.
"""

def create_quad_config(
    mass: float,
    inertia: tuple[float, float, float],
    arm_length: float,
    drag_coeff: float,
    max_rpm: float,
    motor_tau: float,
    hover_rpm_fraction: float = 0.5,
    kf_km_ratio: float =0.02,

) -> QuadConfig:

    rotors = create_quad_rotors(
        arm_length=arm_length,
        max_rpm=max_rpm,
        motor_tau=motor_tau,
        kf_km_ratio=kf_km_ratio,
        hover_rpm_fraction=hover_rpm_fraction,
        mass=mass
    )

    return QuadConfig(
        mass=mass,
        inertia=inertia,
        arm_length=arm_length,
        drag_coeff=drag_coeff,
        rotors=rotors,
    )



"""
Builds a starting QuadState — spawns at rest, level orientation,
unless overridden.
"""
def create_initial_state(
    position: Vector3D,
    velocity: Vector3D | None = None,
    orientation: Quaternion | None = None,
) -> QuadState:

    return QuadState(
        position=position,
        velocity=velocity or Vector3D(0.0, 0.0, 0.0),
        orientation=orientation or Quaternion(0.0, 0.0, 0.0, 1.0),  # identity quaternion
        angular_velocity=Vector3D(0.0, 0.0, 0.0),
        rotor_rpm=[0.0, 0.0, 0.0, 0.0],
    )