"""
Dynamics: the drone itself (rotor/body config, true state) and the only code allowed
to mutate that state (mixer inversion, motor lag, thrust/torque integration).

Per rotor: F_i = k_f * omega_i^2 (thrust, body z-axis), M_i = k_m * omega_i^2 (reaction
torque). The mixer matrix converts the 4 rotor thrusts into total thrust + 3 body
torques and is invertible, so a desired [thrust, roll, pitch, yaw] command inverts back
to the 4 required rotor thrusts. Rotational dynamics include the omega x (I omega)
gyroscopic coupling term, so roll/pitch rates interact even with no torque applied --
they can't be treated as independent linear channels.
"""
from dataclasses import dataclass
