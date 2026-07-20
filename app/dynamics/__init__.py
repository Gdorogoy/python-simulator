"""
dynamics:
the drone itself (rotor/body config, true state) and the only code allowed to mutate that true state:
    mixer inversion, motor lag, thrust/torque integration.

Per rotor, each tick:


F_i = k_f * ω_i²      # thrust, along body z-axis
M_i = k_m * ω_i²       # reaction torque about body z-axis
Mixer matrix — converts the 4 rotor thrusts into total thrust + 3 body torques. This is your linear algebra:


[T ]   [ 1      1      1      1  ] [F1]
[τx] = [ 0      L      0     -L  ] [F2]
[τy] = [-L      0      L      0  ] [F3]
[τz]   [-k_m/k_f  k_m/k_f  -k_m/k_f  k_m/k_f] [F4]
(exact signs depend on your + vs X rotor layout — this is a real design decision, and it's the thing your team lead will care about). This matrix is invertible, which is also how you go the other direction: given a desired thrust+torque command from your RL policy, invert the matrix to get the 4 required rotor thrusts.

Translational dynamics (world frame) — linear, but involves a rotation matrix from body→world:


m * dv/dt = R(orientation) @ [0, 0, T] - [0, 0, m*g] - drag(v) + wind_force
Rotational dynamics (body frame) — this is the part that is not linear:


I * dω/dt = τ - ω × (I ω)
That ω × (Iω) cross-product term is the gyroscopic coupling — pitch and roll rates interact even with no torque applied. It's a small correction at low speed but real, and it's the reason you can't just treat rotation as 3 independent linear channels.

Orientation update — integrate ω into the quaternion (or Euler angles, but Euler has gimbal lock at ±90° pitch, which a school project can usually accept as an edge case).



"""
from dataclasses import dataclass
