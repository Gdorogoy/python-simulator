"""
PyBullet-specific glue code — connects pure dynamics (entity.py, methods.py)
to the actual PyBullet simulation. No physics math lives here, only
PyBullet API calls and translation between our data structures and PyBullet's.
"""
import pybullet as p

from app.dynamics.drone import QuadConfig, Vector3D


def spawn_drone(config: QuadConfig, start_position: Vector3D) -> int:
    """
    Creates a PyBullet rigid body matching the drone's mass/inertia.
    Returns the body_id PyBullet uses to reference this object.
    """

    # creating basic shape for the drone body
    body_half_extents = [config.arm_length * 0.3, config.arm_length * 0.3, 0.05]
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=body_half_extents)
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=body_half_extents, rgbaColor=[1, 0, 0, 1])

    # creating basic shape for the drone rotors and then attaching them to the
    # body model as child links
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
        baseMass=config.mass,                     # the base body mass
        baseCollisionShapeIndex=col_shape,          # the base body's invisible collision shape
        baseVisualShapeIndex=vis_shape,              # the base body model visualized
        basePosition=[start_position.x, start_position.y, start_position.z],  # starting position, absolute world coords

        linkMasses=link_masses,                        # the mass of each linked (attached) rotor
        linkCollisionShapeIndices=link_collision_shapes, # the collision shape of each linked rotor
        linkVisualShapeIndices=link_visual_shapes,        # the model visualized for each linked rotor

        linkPositions=link_positions,                       # WHERE each rotor ATTACHES, relative to the parent (base body)
        linkOrientations=link_orientations,                  # the ROTATION of each rotor's attachment point, relative to parent (quaternion)

        linkInertialFramePositions=link_inertial_positions,   # WHERE each rotor's own MASS is centered, relative to its OWN attachment point
        linkInertialFrameOrientations=link_inertial_orientations, # the ROTATION of each rotor's own mass center, relative to its OWN attachment point

        linkParentIndices=link_parent_indices,                  # which body each rotor is attached to (0 = base body)
        linkJointTypes=link_joint_types,                          # HOW each rotor is allowed to MOVE relative to its parent (FIXED = rigid, no independent motion)
        linkJointAxis=link_joint_axes,                              # rotation/slide axis, only relevant for non-fixed joints (irrelevant here since we use FIXED)
    )

    return drone_id




def apply_rotor_thrust(body_id: int, config: QuadConfig, rotor_speeds: list[float]) -> None:
    """
    Converts 4 rotor speeds into total thrust, applies as a force
    on the PyBullet body along its own body-frame z-axis.
    """


    pass