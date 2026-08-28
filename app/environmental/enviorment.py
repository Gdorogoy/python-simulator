"""
PyBullet-specific glue code — connects pure dynamics (entity.py, methods.py)
to the actual PyBullet simulation. No physics math lives here, only
PyBullet API calls and translation between our data structures and PyBullet's.
"""
import pybullet as p

from app.dynamics.drone import QuadConfig, Vector3D
from app.dynamics.methods import net_combining_thrust, net_combining_torque

def spawn_drone(config: QuadConfig, start_position: Vector3D) -> int:
    """
    Creates a PyBullet rigid body matching the drone's mass/inertia.
    Returns the body_id PyBullet uses to reference this object.
    """

    body_half_extents = [config.arm_length * 0.3, config.arm_length * 0.3, 0.05]
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half_extents)
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half_extents, rgbaColor=[1, 0, 0, 1])

    # Rotors are attached to the body as fixed child links.
    rotor_radius = 0.02
    link_masses = []
    link_collision_shapes = []
    link_visual_shapes = []
    link_positions = []
    link_orientations = []
    link_inertial_positions = []
    link_inertial_orientations = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axes = []

    for rotor in config.rotors:
        rotor_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=rotor_radius,
            rgbaColor=[0, 0, 0, 1]
        )
        rotor_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)  # negligible collision

        link_masses.append(0.01)  # small, mostly cosmetic
        link_collision_shapes.append(rotor_col)
        link_visual_shapes.append(rotor_vis)
        link_positions.append([rotor.position.x, rotor.position.y, rotor.position.z])
        link_orientations.append([0, 0, 0, 1])
        link_inertial_positions.append([0, 0, 0])
        link_inertial_orientations.append([0, 0, 0, 1])
        link_parent_indices.append(0)  # attach to base body
        link_joint_types.append(p.JOINT_FIXED)  # rigidly fixed, doesn't move independently
        link_joint_axes.append([0, 0, 0])

    drone_id = p.createMultiBody(
        baseMass=config.mass,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=[start_position.x, start_position.y, start_position.z],  # absolute world coords

        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shapes,
        linkVisualShapeIndices=link_visual_shapes,

        linkPositions=link_positions,             # where each rotor attaches, relative to the base body
        linkOrientations=link_orientations,        # rotation of each attachment point, relative to parent (quaternion)

        linkInertialFramePositions=link_inertial_positions,        # where each rotor's own mass is centered, relative to its attachment point
        linkInertialFrameOrientations=link_inertial_orientations,   # rotation of that mass center, relative to its attachment point

        linkParentIndices=link_parent_indices,   # which body each rotor is attached to (0 = base body)
        linkJointTypes=link_joint_types,          # FIXED = rigid, no independent motion
        linkJointAxis=link_joint_axes,             # only relevant for non-fixed joints (unused here)
    )

    return drone_id




def sample_wind_conditions(np_rand):
    rng=np_rand
    wind_vector = rng.uniform(-2.0, 2.0, size=3)
    mass_scale = rng.uniform(0.92, 1.08)
    return wind_vector, mass_scale


def apply_rotor_thrust(body_id: int, config: QuadConfig, rotor_speeds: list[float]) -> None:
    """
    Converts 4 rotor speeds into total thrust, applies as a force
    on the PyBullet body along its own body-frame z-axis.
    """
    F=net_combining_thrust(config, rotor_speeds)
    p.applyExternalForce(
        objectUniqueId=body_id,
        linkIndex=-1, # -1 = apply to the base body, not a specific link
        forceObj=[0,0,F], # thrust always points along body's own z-axis
        posObj=[0,0,0], # applied at body's own center
        flags=p.LINK_FRAME, # PyBullet transforms body-frame force into world-frame automatically

    )



def apply_rotor_torque(body_id: int, config: QuadConfig,rotor_speeds: list[float]) -> None:

    F=net_combining_torque(config, rotor_speeds)
    p.applyExternalTorque(
        objectUniqueId=body_id,
        linkIndex=-1,
        torqueObj=list(F),
        flags=p.LINK_FRAME,
    )

