"""
mixer inversion, motor lag, thrust/torque integration.
"""
import numpy as np
import pybullet
from numpy.f2py.rules import rout_rules
from scipy.spatial.transform import Rotation

from app.dynamics.drone import QuadConfig, Vector3D, QuadState, Quaternion

"""
we get the desired [thrust, roll, pitch, yaw]  from the rl, 
we build a method that will by multiplying it against our vector of the drone rotors w 
will return to use the desired [thrust, roll, pitch, yaw] .
    
the m matrix built like this:

for each rotor theres a column that represent each of the output elements.
so in our case:
row 1 (k_f) [ F contribution of 1st rotor , F contribution of 2nd rotor , F contribution of 3rd rotor , F contribution of 4th rotor]
row 2 (y*k_f) [ roll contribution of the 1st rotor , roll contribution of the 2nd rotor , roll contribution of the 3rd rotor , roll contribution of the 4th rotor]
row 3 (-x*k_f) [ pitch contribution of the 1st rotor , pitch contribution of the 2nd rotor , pitch contribution of the 3rd rotor , pitch contribution of the 4th rotor]
row 4 (k_m*spin_dir) [ yaw contribution of the 1st rotor , yaw contribution of the 2nd rotor , yaw contribution of the 3rd rotor , yaw contribution of the 4th rotor]

"""
def mixer(config: QuadConfig, rotors_speed : list[float]):
    M=np.zeros((4,4))
    for i in range(0,4):
        for j in range(0,4):
            if i==0:
                M[i, j] = config.rotors[j].k_f
            if i==1:
                M[i, j] = config.rotors[j].position.y*config.rotors[j].k_f
            if i==2:
                M[i, j] = -(config.rotors[j].position.x) * config.rotors[j].k_f
            if i==3:
                M[i, j] =  config.rotors[j].k_m * config.rotors[j].spin_dir

    res=M @ rotors_speed


    # print("MIXER CALCULATIONS ========M===============")
    # print(M)
    # print("MIXER CALCULATIONS =======RES===============")
    # print(res)
    return res


"""
we have the method that will give us by multiplication the desried [thrust, roll, pitch, yaw] ,
but the drone dosent undertand waht to do with it so we do inversion on the smae method to get the insturcions.
in other words what w,rpm,inertia,cw/ccw etc to set the SPECIFIC rotor
"""
def mixer_inversion(config: QuadConfig, desired_params : list[float]) -> list[float]:
    M = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            if i == 0:
                M[i, j] = config.rotors[j].k_f
            if i == 1:
                M[i, j] = config.rotors[j].position.y * config.rotors[j].k_f
            if i == 2:
                M[i, j] = -(config.rotors[j].position.x) * config.rotors[j].k_f
            if i == 3:
                M[i, j] = config.rotors[j].k_m * config.rotors[j].spin_dir

    speeds=np.linalg.inv(M) @ desired_params
    speeds = np.clip(speeds, 0.0, None)
    res = [np.sqrt(speed) for speed in speeds]
    #
    # print("MIXER CALCULATIONS ========M===============")
    # print(np.linalg.inv(M))
    # print("MIXER CALCULATIONS =======RES===============")
    # print(res)

    return res

"""
small_tau= responsive , big_tau = unresponsive
how many seconds to get to wanted rpm

adds realism , like an car engine cant go from 0 to 100kmh in 1 sec 
so the drone rotators also needs to follow realism , 

then calculating the change of the omega via dt, like the velocity change
"""
def motor_lag(w_current: float, w_target: float, motor_tau: float, dt: float) -> float:
    w_dot= (w_target-w_current) / motor_tau
    w_new = w_current + w_dot * dt
    return w_new


"""
each rotator produces 2 forces:
one thats pushing the drone in the direction
another one that twists the drone body (aka tilting it),
we need to combine all 4 of the rotators to get the 

thrust =z index (up)

F_i= thrust magnitude produced by the rotator (always POSITIVE!)
w_i= angular speed of rotator always , because squared the thrust dosent care whether the rotator spins CW or CCW
k_f= thrust coefficient , constant absorbing air density , propeller disk , blade pitch/shape.

no spin_dir beacuse thrust can either go up or down.
"""

def thrust(k_f: float, w_i: float):
    F_i=k_f*w_i**2
    return F_i

"""
torque=rotation 

M_i=reaction about the vertcial axis contributed by rotator i
w_i= same speeds-squared as thrust, more speed more drag on the blades
k_m= torque coefficient ,along the rotational drag
spin_dir=usually -1 or +1 CW=1 CCW=-1 (or vice versa) depends on the convention
 
spin_dir exist because unlinke in thrust the torque directions depends on the spin direction,
 
"""

def torque(k_m: float, w_i: float, spin_dir_i: float):
    F_i= k_m*w_i**2 * spin_dir_i
    return F_i


"""
combining all of the thrust that we calculated per rotor to apply it to the body
   thrust = simple sum

"""

def net_combining_thrust(config: QuadConfig, rotor_speeds: list[float]):
    F_net=np.zeros(4)
    for i in range(0,4):
        F_net[i]=thrust(config.rotors[i].k_f, rotor_speeds[i])

    return sum(F_net)
"""
combining all of the torque that we calculated per rotor to apply it to the body
torque = reaction sum + cross product sum
"""
def net_combining_torque(config: QuadConfig, rotor_speeds: list[float]):
    F_net = np.zeros(3)

    for i in range(0, 4):
        rotor = config.rotors[i]
        F_i = thrust(rotor.k_f, rotor_speeds[i])
        r_i = np.array([rotor.position.x, rotor.position.y, rotor.position.z])
        F_vec = np.array([0, 0, F_i])
        F_net += np.cross(r_i, F_vec)  # moment-arm contribution
        F_net += np.array([0, 0, torque(rotor.k_m, rotor_speeds[i], rotor.spin_dir)])  # reaction torque
    return F_net

"""
how we accelerate on each of our axis (roll , yaw ,pitch)
"""
def angular_acceleration(config: QuadConfig, net_torque: list[float]) -> np.ndarray:
    alpha = np.zeros(3)
    for i in range(3):
        alpha[i] = net_torque[i] / config.inertia[i]
    return alpha


"""
updating the velocity by adding to it the alpha (angular_acceleration) times dt (the time change)
"""
def update_angular_velocity(state: QuadState, alpha: np.ndarray, dt: float) -> np.ndarray:
    current_w = np.array([state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z])
    new_w = current_w + alpha * dt
    return new_w



"""
input: drones current velocity (and in what direction)
output:  the forces that are on the opposite of the direction of the drone on each of the axis,
          that will be used in thrust/gravity/wind , and in acceleration calculations
"""
def drag_force(velocity: list[float] , drag_coeff : float , cross_sec_area : float , air_dens : float):
    v=np.array(velocity)
    speed=np.linalg.norm(v)

    if speed<0.01:
        return np.zeros(3)

    F_drag= 0.5* air_dens * speed**2 * drag_coeff * cross_sec_area

    return (-v/speed) *  F_drag


"""
input: wind vector with direction and magnitude as it elements , mass of the drone , wind coefficient
output: an wind force vecotr on each of the axis that will be added in thrust/drag/gravity
"""
def wind(wind: list[float] , mass: float , k_wind_coeff : float):
    F_wind = np.zeros(3)
    for i in range(0,3):
        F_wind[i]=wind[i] * mass * k_wind_coeff

    return F_wind




def timestamp_update(state: QuadState, config: QuadConfig , rl_action: list[float] , wind_vector : list[float] ,dt: float):

    # print(f"=====================================================================")

    w_target : list[float] = mixer_inversion(config, rl_action)

    # print(f"w_target: {w_target}")



    w_actual = np.zeros(4)
    for i in range(0,4):
        w_actual[i] = motor_lag(state.rotor_rpm[i], w_target[i], config.rotors[i].motor_tau, dt)

    # print(f"w_actual: {w_actual}")

    net_thrust = net_combining_thrust(config, w_actual)  # scalar
    net_torque = net_combining_torque(config, w_actual)  # 3-element vector

    # print(f"net_thrust: {net_thrust}")
    # print(f"net_torque: {net_torque}")

    drag = drag_force(
        velocity=[state.velocity.x, state.velocity.y, state.velocity.z],
        drag_coeff=config.drag_coeff,
        cross_sec_area=0.05,  # you'll want this as a proper QuadConfig field eventually
        air_dens=1.225,
    )

    # print(f"drag: {drag}")

    wind_force = wind(
        wind=wind_vector,  # the function parameter you already receive
        mass=config.mass,
        k_wind_coeff=0.1,  # tunable constant
    )

    # print(f"wind_force: {wind_force}")

    gravity_force = np.array([0, 0, -config.mass * 9.81])

    # print(f"gravity_force: {gravity_force}")

    current_rot = Rotation.from_quat(
        [state.orientation.x, state.orientation.y, state.orientation.z, state.orientation.w])

    # print(f"current_rot: {current_rot}")

    thrust_body_frame = np.array([0, 0, net_thrust])

    # print(f"thrust_body_frame: {thrust_body_frame}")

    thrust_world_frame = current_rot.apply(thrust_body_frame)

    # print(f"thrust_world_frame: {thrust_world_frame}")

    total_force = thrust_world_frame + drag + wind_force + gravity_force

    # print(f"total_force: {total_force}")

    linear_accel = total_force / config.mass

    # print(f"linear_accel: {linear_accel}")

    new_velocity = np.array([state.velocity.x, state.velocity.y, state.velocity.z]) + linear_accel * dt

    # print(f"new_velocity: {new_velocity}")


    new_position = np.array([state.position.x, state.position.y, state.position.z]) + new_velocity * dt

    # print(f"new_position: {new_position}")

    alpha = angular_acceleration(config, net_torque)

    # print(f"alpha: {alpha}")

    new_angular_velocity = update_angular_velocity(state, alpha, dt)

    # print(f"new_angular_velocity: {new_angular_velocity}")

    delta_rot = Rotation.from_rotvec(new_angular_velocity * dt)

    # print(f"delta_rot: {delta_rot}")

    new_rot = delta_rot * current_rot

    # print(f"new_rot: {new_rot}")




    new_quat = new_rot.as_quat()  # returns [x, y, z, w]


    # print(f"new_quat value: {new_quat}")

    quad_state=QuadState(
        position=Vector3D(*new_position),
        velocity=Vector3D(*new_velocity),
        orientation=Quaternion(*new_quat),
        angular_velocity=Vector3D(*new_angular_velocity),
        rotor_rpm=list(w_actual),
    )

    return quad_state
