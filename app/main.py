import time
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion
from app.dynamics.methods import timestamp_update, mixer_inversion
from app.environmental.enviorment import spawn_drone


# ---------------------------------------------------------------------------
# Controllers (all take the CURRENT state, return a torque/thrust command)
# ---------------------------------------------------------------------------

def altitude_thrust(state, target_z, hover_thrust, current_roll=0.0, current_pitch=0.0,
                     kp=15.0, kd=8.0):
    """
    P-D controller on altitude, PLUS a feed-forward tilt compensation term.
    When banked, only cos(roll)*cos(pitch) of total thrust actually points
    "up" — this pre-emptively boosts thrust so altitude doesn't sag while tilted.
    """
    error = target_z - state.position.z
    vz = state.velocity.z
    base = hover_thrust + kp * error - kd * vz
    tilt_factor = max(np.cos(current_roll) * np.cos(current_pitch), 0.5)
    return base / tilt_factor


def attitude_torque(state, target_roll, target_pitch, target_yaw_rate=0.0,
                     kp=8.0, kd=1.5, kd_yaw=1.5):
    """
    Inner-loop attitude controller. Drives current roll/pitch toward the
    given TARGET ANGLES (not toward zero) — lets you hold a bank angle,
    and (via target_yaw_rate) spin at a controlled rate.
    """
    rot = Rotation.from_quat([state.orientation.x, state.orientation.y,
                               state.orientation.z, state.orientation.w])
    roll, pitch, yaw = rot.as_euler('xyz')
    w = state.angular_velocity

    roll_t  = -kp * (roll - target_roll)   - kd * w.x
    pitch_t = -kp * (pitch - target_pitch) - kd * w.y
    yaw_t   = -kd_yaw * (w.z - target_yaw_rate)
    return roll_t, pitch_t, yaw_t


def position_to_attitude(state, target_x, target_y, kp=0.12, kd=0.22, max_tilt=0.35):
    """
    Outer-loop position controller. Converts a world-frame XY position
    error into small desired roll/pitch ANGLES. Used for translate/circle/
    wait/return — anything that needs to reach or hold an actual (x,y).
    """
    ex = target_x - state.position.x
    ey = target_y - state.position.y
    vx, vy = state.velocity.x, state.velocity.y

    accel_x_desired = kp * ex - kd * vx
    accel_y_desired = kp * ey - kd * vy

    # pitch forward (+) -> accelerates +x ; roll (+) -> accelerates -y
    # (sign convention matches this project's body-frame axes — flip signs
    # here first if you see it drive the wrong way on your rig)
    target_pitch = np.clip(accel_x_desired, -max_tilt, max_tilt)
    target_roll  = np.clip(-accel_y_desired, -max_tilt, max_tilt)
    return target_roll, target_pitch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # timestamp_update() applies gravity itself
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=45, cameraPitch=-30,
                                  cameraTargetPosition=[0, 0, 5])

    config = create_quad_config(
        mass=1.5, inertia=(0.02, 0.02, 0.04), arm_length=0.15,
        drag_coeff=0.02, max_rpm=800, motor_tau=0.05,
    )

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
    steps_per_control = max(1, round((1 / control_hz) / physics_dt))  # -> 3
    wind_vector = [0.0, 0.0, 0.0]

    BANK_ANGLE = 0.25          # ~14 degrees, used for tilt_hold phases
    YAW_SPIN_RATE = 2 * np.pi / 3.0  # rad/s -> one full 360 deg turn in 3s

    # -----------------------------------------------------------------
    # Phase modes:
    #   "altitude"   -> move to target_z, x/y held at 0
    #   "wait"       -> freeze wherever it currently is (captured at phase start)
    #   "tilt_hold"  -> hold a fixed roll/pitch (will drift a little — real banking)
    #   "translate"  -> fly to an explicit (x,y) target
    #   "circle"     -> track a circle of given radius around origin
    #   "yaw_spin"   -> spin in place at a fixed yaw rate, level roll/pitch
    # -----------------------------------------------------------------
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

        # "wait" and "yaw_spin" both freeze wherever the drone currently is,
        # captured ONCE here at the start of the phase
        wait_x, wait_y = state.position.x, state.position.y
        spin_x, spin_y = state.position.x, state.position.y

        for c in range(n_control_steps):
            t = c / control_hz  # time since this phase started

            rot_now = Rotation.from_quat([state.orientation.x, state.orientation.y,
                                          state.orientation.z, state.orientation.w])
            cur_roll, cur_pitch, _ = rot_now.as_euler('xyz')

            if phase["mode"] == "altitude":
                thrust = altitude_thrust(state, phase["target_z"], hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(state, 0.0, 0.0)
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "wait":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(state, wait_x, wait_y)
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "tilt_hold":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                roll_t, pitch_t, yaw_t = attitude_torque(state, phase["roll"], phase["pitch"])

            elif phase["mode"] == "translate":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(
                    state, phase["target_x"], phase["target_y"])
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "circle":
                omega_orbit = 2 * np.pi / phase["period"]
                tx = phase["radius"] * np.cos(omega_orbit * t)
                ty = phase["radius"] * np.sin(omega_orbit * t)
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(state, tx, ty, kp=0.20, kd=0.28)
                roll_t, pitch_t, yaw_t = attitude_torque(state, target_roll, target_pitch)

            elif phase["mode"] == "yaw_spin":
                thrust = altitude_thrust(state, start_z, hover_thrust, cur_roll, cur_pitch)
                target_roll, target_pitch = position_to_attitude(state, spin_x, spin_y)
                roll_t, pitch_t, yaw_t = attitude_torque(
                    state, target_roll, target_pitch, target_yaw_rate=phase["yaw_rate"])

            action = [thrust, roll_t, pitch_t, yaw_t]

            # ---- apply that SAME action for steps_per_control physics ticks ----
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

            if c % control_hz == 0:  # print once per second of sim time
                print(f"  t={t:5.1f}s  pos=({state.position.x:6.2f}, "
                      f"{state.position.y:6.2f}, {state.position.z:6.2f})")

    print("\nFlight demo complete. Disconnecting.")
    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    main()