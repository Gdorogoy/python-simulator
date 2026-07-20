
import pybullet as p
import pybullet_data
import time

from app.dynamics.drone import create_quad_config, Vector3D
from app.environmental.enviorment import spawn_drone


def main():
    # --- connect to PyBullet with a visible GUI window ---
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")  # ground plane, so you can see relative altitude

    # --- build a test drone config ---
    config = create_quad_config(
        mass=1.5,
        inertia=(0.02, 0.02, 0.04),
        arm_length=0.15,
        drag_coeff=0.02,
        max_rpm=800,
        motor_tau=0.05,
    )


    start_position = Vector3D(0, 0, 3.0)
    drone_id = spawn_drone(config, start_position)

    print(f"Drone spawned successfully. body_id = {drone_id}")

    # --- confirm spawn position/mass match what we passed in ---
    pos, orn = p.getBasePositionAndOrientation(drone_id)
    dynamics_info = p.getDynamicsInfo(drone_id, -1)
    print(f"Spawned position: {pos}")
    print(f"Spawned orientation (quaternion): {orn}")
    print(f"Reported mass: {dynamics_info[0]}  (should match config.mass = {config.mass})")

    # --- let it just sit there under gravity for a few seconds, no thrust yet ---
    # it should FALL, since no force is being applied to counter gravity —
    # this alone confirms mass/gravity/collision are all working correctly
    for step in range(300):
        p.stepSimulation()
        if step % 50 == 0:
            pos, _ = p.getBasePositionAndOrientation(drone_id)
            print(f"step {step}: position = {pos}")
        time.sleep(1 / 240)

    print("Test complete. Close the PyBullet window to exit.")
    input("Press Enter to disconnect...")
    p.disconnect()


if __name__ == "__main__":
    main()