import time
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion
from app.dynamics.methods import timestamp_update, mixer_inversion
from app.environmental.enviorment import spawn_drone, sample_wind_conditions


# ---------------------------------------------------------------------------
# Controllers (all take the CURRENT state, return a torque/thrust command)
# ---------------------------------------------------------------------------

def angle_diff(a, b):
    """Shortest signed difference between two angles, wrapped to [-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def altitude_thrust(state, target_z, hover_thrust, current_roll=0.0, current_pitch=0.0,
                     kp=15.0, kd=8.0):
    error = target_z - state.position.z
    vz = state.velocity.z
    base = hover_thrust + kp * error - kd * vz
    tilt_factor = max(np.cos(current_roll) * np.cos(current_pitch), 0.5)
    return base / tilt_factor


def attitude_torque(state, target_roll, target_pitch, target_yaw_rate=0.0,
                     kp=8.0, kd=1.5, kd_yaw=1.5):
    """
    Inner-loop attitude controller. Computes the error as a QUATERNION
    rotation between current and target attitude, NOT by subtracting
    Euler angles. Euler subtraction (roll - target_roll) gets corrupted
    by roll/pitch/yaw coupling whenever yaw is spinning fast — this is
    exactly what caused the crash during "yaw_spin": the earlier
    wraparound fix only handled the +/-pi discontinuity, not the deeper
    coupling that exists even mid-range. The quaternion error below never
    subtracts roll/pitch numbers at all, so it doesn't inherit that bug.
    """
    cur_rot = Rotation.from_quat([state.orientation.x, state.orientation.y,
                                   state.orientation.z, state.orientation.w])
    # only extract yaw (to build a target at the SAME heading) — never
    # extract/subtract roll or pitch, that's the part that was breaking
    _, _, cur_yaw = cur_rot.as_euler('xyz')
    target_rot = Rotation.from_euler('xyz', [target_roll, target_pitch, cur_yaw])

    # rotation FROM current TO target, expressed in the drone's own body
    # frame — this lines up directly with roll/pitch torque axes
    error_rot = cur_rot.inv() * target_rot
    error_vec = error_rot.as_rotvec()

    w = state.angular_velocity
    roll_t  = kp * error_vec[0] - kd * w.x
    pitch_t = kp * error_vec[1] - kd * w.y
    yaw_t   = -kd_yaw * (w.z - target_yaw_rate)
    return roll_t, pitch_t, yaw_t


def position_to_attitude(state, target_x, target_y,current_yaw=0.0, kp=0.12, kd=0.22, max_tilt=0.35):
    ex = target_x - state.position.x
    ey = target_y - state.position.y
    vx, vy = state.velocity.x, state.velocity.y

    accel_x_world = kp * ex - kd * vx
    accel_y_world = kp * ey - kd * vy

    # rotate desired WORLD-frame acceleration into BODY-frame using current yaw
    cos_y, sin_y = np.cos(current_yaw), np.sin(current_yaw)
    accel_x_body = accel_x_world * cos_y + accel_y_world * sin_y
    accel_y_body = -accel_x_world * sin_y + accel_y_world * cos_y

    target_pitch = np.clip(accel_x_body, -max_tilt, max_tilt)
    target_roll  = np.clip(-accel_y_body, -max_tilt, max_tilt)
    return target_roll, target_pitch


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30,
                                  cameraTargetPosition=[0, 0, 5])

    wind_vector, mass_scale = sample_wind_conditions(np.random)

    config = create_quad_config(
        mass=1.5 * mass_scale,
        inertia=(0.02 * mass_scale, 0.02 * mass_scale, 0.04 * mass_scale),
        arm_length=0.22,
        drag_coeff=0.035,
        max_rpm=12000,
        motor_tau=0.05,
    )
    print(f"config: {config}")

    start_z = 5.0
    start_position = Vector3D(0, 0, start_z)
    drone_id = spawn_drone(config, start_position)

    hover_thrust = config.mass * 9.81
    hover_omega = mixer_inversion(config, [hover_thrust, 0.0, 0.0, 0.0])

    state = QuadState(
        position=start_position,
        velocity=Vector3D(0, 0, 0),
        orientation=Quaternion(0, 0, 0, 1),
        angular_velocity=Vector3D(0, 0, 0),
        rotor_rpm=list(hover_omega),
    )

    physics_dt = 1 / 240
    control_hz = 75
    steps_per_control = max(1, round((1 / control_hz) / physics_dt))

    BANK_ANGLE = 0.25
    YAW_SPIN_RATE = (2 * np.pi / 3.0 ) *2.5

    flight_plan = [
        dict(name="1. Descend 2m",            mode="altitude", target_z=start_z - 2, duration=3.0),
        dict(name="1. Wait 3s",               mode="wait",     duration=3.0),

        dict(name="2. Ascend back 2m",        mode="altitude", target_z=start_z,     duration=3.0),
        dict(name="2. Wait 3s",               mode="wait",     duration=3.0),

        dict(name="3. Tilt left, hold 3s",    mode="tilt_hold", roll=+BANK_ANGLE, pitch=0.0, duration=3.0),

        dict(name="4. Go left 2m",            mode="translate", target_x=0.0, target_y=-2.0, duration=3.0),
        dict(name="4. Go back to y=0",        mode="translate", target_x=0.0, target_y=0.0,  duration=3.0),

        dict(name="5. Wait 3s",               mode="wait",     duration=3.0),

        dict(name="6. Tilt right, hold 3s",   mode="tilt_hold", roll=-BANK_ANGLE, pitch=0.0, duration=3.0),

        dict(name="7. Go right 2m",           mode="translate", target_x=0.0, target_y=2.0, duration=3.0),
        dict(name="7. Go back to y=0",        mode="translate", target_x=0.0, target_y=0.0, duration=3.0),

        dict(name="8. Wait 3s",               mode="wait",     duration=3.0),

        dict(name="9. Circle r=2m",           mode="circle",   radius=2.0, period=10.0, duration=10.0),

        dict(name="10. Full yaw spin (360)",  mode="yaw_spin", yaw_rate=YAW_SPIN_RATE, duration=3.0),

        dict(name="11. Finish - final hover", mode="wait",     duration=2.0),
    ]

    print("Starting choreographed flight. Control @ %dHz, physics @ %dHz "
          "(%d physics ticks per control step)." % (control_hz, 1/physics_dt, steps_per_control))

    step_count = 0

    for phase in flight_plan:
        print(f"\n--- Phase: {phase['name']} ---")
        n_control_steps = int(phase["duration"] * control_hz)

        hover_thrust_needed = config.mass * 9.81
        w_hover = np.sqrt((hover_thrust_needed / 4) / config.rotors[0].k_f)
        print(f"current w_hover{w_hover}")

        wait_x, wait_y = state.position.x, state.position.y
        spin_x, spin_y = state.position.x, state.position.y

        for c in range(n_control_steps):
            t = c / control_hz

            rot_now = Rotation.from_quat([state.orientation.x, state.orientation.y,
                                          state.orientation.z, state.orientation.w])
            cur_roll, cur_pitch,cur_yaw  = rot_now.as_euler('xyz')

            if phase["mode"] == "altitude":
                thrust = altitude_thrust(state, phase["target_z"], hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(state, 0.0, 0.0, current_yaw=cur_yaw)
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "wait":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(state, wait_x, wait_y, current_yaw=cur_yaw)
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "tilt_hold":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                roll_t, pitch_t, yaw_t = attitude_torque(state, phase["roll"], phase["pitch"])

            elif phase["mode"] == "translate":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(
                    state, phase["target_x"], phase["target_y"], current_yaw=cur_yaw)
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "circle":
                omega_orbit = 2 * np.pi / phase["period"]
                tx = phase["radius"] * np.cos(omega_orbit * t)
                ty = phase["radius"] * np.sin(omega_orbit * t)
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch,)
                target_roll, target_pitch = position_to_attitude(state, tx, ty, kp=0.20, kd=0.28, current_yaw=cur_yaw )
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "yaw_spin":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(state, spin_x, spin_y, current_yaw=cur_yaw,)
                roll_t, pitch_t, yaw_t = attitude_torque(
                    state, target_roll, target_pitch, target_yaw_rate=phase["yaw_rate"])

            action = [thrust, roll_t, pitch_t, yaw_t]

            for _ in range(steps_per_control):
                if not p.isConnected():
                    print("\nPyBullet window was closed — stopping cleanly.")
                    return

                state = timestamp_update(state, config, action, wind_vector, physics_dt)

                pos = np.array([state.position.x, state.position.y, state.position.z])
                if np.any(np.isnan(pos)):
                    print("!! NaN position — stopping.")
                    p.disconnect(); return
                if pos[2] < 0.0:
                    print("!! Drone hit the ground.")
                    p.disconnect(); return
                if np.linalg.norm(pos[:2]) > 50.0:
                    print("!! Drone flew outside the 50m safe area — stopping.")
                    p.disconnect(); return

                try:
                    p.resetBasePositionAndOrientation(
                        drone_id,
                        [state.position.x, state.position.y, state.position.z],
                        [state.orientation.x, state.orientation.y,
                         state.orientation.z, state.orientation.w],
                    )
                    p.stepSimulation()
                except p.error:
                    print("\nPyBullet window was closed — stopping cleanly.")
                    return

                time.sleep(physics_dt)
                step_count += 1

            if c % control_hz == 0:
                print(f"  t={t:5.1f}s  pos=({state.position.x:6.2f}, "
                      f"{state.position.y:6.2f}, {state.position.z:6.2f})")

    print("\nFlight demo complete. Disconnecting.")
    if p.isConnected():
        p.disconnect()
