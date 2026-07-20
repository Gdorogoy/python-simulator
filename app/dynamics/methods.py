"""
mixer inversion, motor lag, thrust/torque integration.
"""
import numpy as np
import pybullet

from app.dynamics.drone import QuadConfig, Vector3D

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
                M[i, j] = config.rotors[i].k_f
            if i==1:
                M[i, j] = config.rotors[j].position.y*config.rotors[i].k_f
            if i==2:
                M[i, j] = -(config.rotors[j].position.x) * config.rotors[i].k_f
            if i==3:
                M[i, j] =  config.rotors[i].k_m * config.rotors[j].spin_dir

    res=M @ rotors_speed


    print("MIXER CALCULATIONS ========M===============")
    print(M)
    print("MIXER CALCULATIONS =======RES===============")
    print(res)
    return res


"""
we have the method that will give us by multiplication the desried [thrust, roll, pitch, yaw] ,
but the drone dosent undertand waht to do with it so we do inversion on the smae method to get the insturcions.
in other words what w,rpm,inertia,cw/ccw etc to set the SPECIFIC rotor
"""
def mixer_inversion(config: QuadConfig, desired_params : list[float]):
    M = np.zeros((4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            if i == 0:
                M[i, j] = config.rotors[i].k_f
            if i == 1:
                M[i, j] = config.rotors[j].position.y * config.rotors[i].k_f
            if i == 2:
                M[i, j] = -(config.rotors[j].position.x) * config.rotors[i].k_f
            if i == 3:
                M[i, j] = config.rotors[i].k_m * config.rotors[j].spin_dir

    speeds=np.linalg.inv(M) @ desired_params
    res = [np.sqrt(speed) for speed in speeds]

    print("MIXER CALCULATIONS ========M===============")
    print(np.linalg.inv(M))
    print("MIXER CALCULATIONS =======RES===============")
    print(res)

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
    F_net = np.zeros(4)
    for i in range(0, 4):
        F_net[i] = torque(config.rotors[i].k_m, rotor_speeds[i],config.rotors[i].spin_dir)

    return sum(F_net)





