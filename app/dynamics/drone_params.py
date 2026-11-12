"""The one place the drone's physical parameters and action bounds live.

Everything that builds the plant (numpy env, Isaac env, PID controllers, gain tuner, the viewer) reads these, so the
numpy and Isaac simulators cannot drift apart. No torch / Isaac imports: safe to import from anywhere.

Changing a value here changes the plant or what the policy's outputs MEAN (the action bounds), so checkpoints trained
with other values need retraining. A checkpoint stores its own bounds (action_low / action_high buffers) and the viewer
keeps using them; the env just clips to the bounds below.
"""

# ---- airframe (generic 1.5 kg X-quad; replace with a real airframe's numbers when targeting one) ----
MASS = 1.5                       # kg
INERTIA = (0.02, 0.02, 0.04)     # kg m^2, Ixx Iyy Izz
ARM_LENGTH = 0.22                # m, centre to rotor
MAX_RPM = 12000.0                # rotor speed at which one rotor makes 4x its hover thrust
MOTOR_TAU = 0.05                 # s, first-order rotor spin-up / spin-down lag
GRAVITY = 9.81

# ---- aerodynamics: F = 0.5 * rho * v^2 * DRAG_COEFF * CROSS_SEC_AREA, against the velocity ----
# Was 0.035 (about 0.35 N at 18 m/s, 2% of the weight, so the drone coasted like it was in a vacuum). 0.5 is a bluff body
# of a real quad's size: about 5 N (a third of the weight) at 18 m/s, which also gives a real top speed.
DRAG_COEFF = 0.5
CROSS_SEC_AREA = 0.05            # m^2
AIR_DENS = 1.225                 # kg / m^3

# ---- angle-style action bounds: [thrust above hover (N), roll, pitch, yaw torque (N m)] ----
# Thrust: the action is thrust ABOVE hover. Down to zero thrust (-1 x hover); up to THRUST_UP_FACTOR x hover on top of hover, i.e.
# (1 + THRUST_UP_FACTOR) x weight in total. Was 1.0 (2x weight, +1 g net, and no tilt past 60 deg while holding altitude);
# 2.0 gives 3x weight (+2 g net, altitude held up to 70.5 deg tilt). The rotors themselves reach 4x weight, which leaves
# the rest as headroom for the roll / pitch / yaw differential.
THRUST_UP_FACTOR = 2.0
# Roll / pitch torque. Was 0.5 (25 rad/s^2, 0.5 s to tilt 60 deg); the rotors can make about 4-5 N m. 1.5 gives 75 rad/s^2.
MAX_TORQUE_ROLL_PITCH = 1.5
# Yaw: only about 0.3 N m is reachable at hover (the rotors' drag torque is 2% of their thrust), so 0.5 is already the physical cap.
MAX_TORQUE_YAW = 0.5


def hover_thrust(mass: float = MASS) -> float:
    return mass * GRAVITY


def angle_action_bounds(mass: float = MASS):
    """(low, high) of the angle-style action, as tuples of 4 floats."""
    h = hover_thrust(mass)
    t, y = MAX_TORQUE_ROLL_PITCH, MAX_TORQUE_YAW
    return (-h, -t, -t, -y), (THRUST_UP_FACTOR * h, t, t, y)


def acro_thrust_bounds(mass: float = MASS):
    """(low, high) of the acro action's thrust entry; the rate entries are MAX_BODY_RATE (acro_style_methods)."""
    h = hover_thrust(mass)
    return -h, THRUST_UP_FACTOR * h
