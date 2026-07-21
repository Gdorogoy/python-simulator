"""
Flight demo — cycles through hover, roll, pitch, and yaw phases so you
can see the drone actually tilt and rotate, not just hover or fall.

Includes a basic safety boundary check so it doesn't silently fly off
into nowhere if something is miscalibrated — prints a warning and stops
early instead of running blind.

FIX: rotors now start PRE-SPUN to the exact hover-equilibrium speed,
instead of starting from 0. Starting from 0 meant the drone fell for
the entire ~100-step motor spin-up window before thrust could counter
gravity, building up downward velocity that persisted even once thrust
matched weight (net force ~0 means CONSTANT velocity, not zero velocity).
Also raised starting altitude to give extra margin.
"""
import time
import numpy as np
import pybullet as p
import pybullet_data

from app.dynamics.drone import create_quad_config, QuadState, Vector3D, Quaternion
from app.dynamics.methods import timestamp_update, mixer_inversion
from app.environmental.enviorment import spawn_drone


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # physics_step() handles gravity itself
    p.loadURDF("plane.urdf")

    config = create_quad_config(
        mass=1.5,
        inertia=(0.02, 0.02, 0.04),
        arm_length=0.15,
        drag_coeff=0.02,
        max_rpm=800,
        motor_tau=0.05,
    )

    # raised starting altitude — extra safety margin
    start_position = Vector3D(0, 0, 5.0)
    drone_id = spawn_drone(config, start_position)

    hover_thrust = config.mass * 9.81

    # --- pre-spin rotors to hover-equilibrium speed, instead of starting at 0 ---
    # this simulates a drone that's ALREADY stably hovering when the demo begins,
    # rather than one powering on from a dead stop
    hover_omega = mixer_inversion(config, [hover_thrust, 0.0, 0.0, 0.0])

    state = QuadState(
        position=start_position,
        velocity=Vector3D(0, 0, 0),
        orientation=Quaternion(0, 0, 0, 1),
        angular_velocity=Vector3D(0, 0, 0),
        rotor_rpm=list(hover_omega),
    )

    dt = 1 / 240
    wind_vector = [0.0, 0.0, 0.0]

    # small torque nudges — deliberately conservative so we don't push
    # mixer_inversion into producing negative omega^2 for any rotor

    desired_alpha = 1.5  # rad/s² — gentle angular acceleration
    small_torque = desired_alpha * config.inertia[0]


    # each phase: (name, [F, roll, pitch, yaw], duration_in_steps)
    flight_plan = [
        ("Hover", [hover_thrust, 0.0, 0.0, 0.0], 300),
        ("Roll right", [hover_thrust, small_torque, 0.0, 0.0], 100),
        ("Counter-roll", [hover_thrust, -small_torque, 0.0, 0.0], 100),
        ("Hover (level)", [hover_thrust, 0.0, 0.0, 0.0], 200),
        ("Pitch forward", [hover_thrust, 0.0, small_torque, 0.0], 100),
        ("Counter-pitch", [hover_thrust, 0.0, -small_torque, 0.0], 100),
        ("Hover (level)", [hover_thrust, 0.0, 0.0, 0.0], 200),
        ("Yaw spin", [hover_thrust, 0.0, 0.0, small_torque], 100),
        ("Counter-yaw", [hover_thrust, 0.0, 0.0, -small_torque], 100),
        ("Hover (level, final)", [hover_thrust, 0.0, 0.0, 0.0], 300),
    ]


    print("Starting flight demo. Watch the PyBullet window.")
    print(f"Hover-equilibrium rotor speed: {hover_omega}")

    step_count = 0
    for phase_name, rl_action, duration in flight_plan:
        print(f"\n--- Phase: {phase_name} ---")

        for _ in range(duration):
            state = timestamp_update(state, config, rl_action, wind_vector, dt)

            # --- safety check: stop early if something has clearly gone wrong ---
            pos = np.array([state.position.x, state.position.y, state.position.z])
            if np.any(np.isnan(pos)):
                print("!! Position became NaN — likely negative omega^2 in "
                      "mixer_inversion. Check the clip fix.")
                p.disconnect()
                return
            if pos[2] < 0.0:
                print("!! Drone hit the ground.")
                p.disconnect()
                return
            if np.linalg.norm(pos[:2]) > 50.0:
                print("!! Drone flew outside the safe area (>50m from origin) — "
                      "stopping early rather than let it fly off the map.")
                p.disconnect()
                return

            p.resetBasePositionAndOrientation(
                drone_id,
                [state.position.x, state.position.y, state.position.z],
                [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w],
            )

            if step_count % 100 == 0:
                print(f"step {step_count}: pos=({state.position.x:.2f}, "
                      f"{state.position.y:.2f}, {state.position.z:.2f})  "
                      f"orientation=({state.orientation.x:.3f}, {state.orientation.y:.3f}, "
                      f"{state.orientation.z:.3f}, {state.orientation.w:.3f})")

            p.stepSimulation()
            time.sleep(dt)
            step_count += 1

    print("\nFlight demo complete.")
    input("Press Enter to disconnect...")
    p.disconnect()


if __name__ == "__main__":
    main()