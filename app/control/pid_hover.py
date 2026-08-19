from os import access

import numpy as np
from scipy.spatial.transform import Rotation

class PIDHoverController:
    def __init__(self, kp_pos, kd_pos, kp_att, kd_att, kp_yaw, kd_yaw, max_tilt_rad=0.3):

        #p - proportional gain, how hard to react to current error (how far rn)
        #d - derivative gain, how hard to react to rate of change (prevents overshoot)
        #

        # pos control kp pushes towards the target kd damps based on velocity , output is desired acceleration
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos

        # attitude kp pushes toward desired roll pitch the current roll pitch kd damps based on the acceleration
        self.kp_att = kp_att
        self.kd_att = kd_att

        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.max_tilt_rad = max_tilt_rad

    def compute_action(self, drone_state, target_pos):

        # 1 read state
        pos= np.array([drone_state.position.x, drone_state.position.y, drone_state.position.z])
        vel=np.array([drone_state.velocity.x, drone_state.velocity.y, drone_state.velocity.z])
        ang_vel=np.array([drone_state.angular_velocity.x, drone_state.angular_velocity.y, drone_state.angular_velocity.z])
        roll, pitch, yaw = Rotation.from_quat([
            drone_state.orientation.x, drone_state.orientation.y,
            drone_state.orientation.z, drone_state.orientation.w
        ]).as_euler("xyz")

        pos_err= target_pos - pos

        # 2 outer loop
        accel_cmd= self.kp_pos* pos_err - self.kd_pos* vel

        des_roll=np.clip(-accel_cmd[1]/9.81,-self.max_tilt_rad,self.max_tilt_rad)
        des_pitch=np.clip(accel_cmd[0]/9.81,-self.max_tilt_rad,self.max_tilt_rad)

        # 3 inner loop
        roll_torque= self.kp_att* (des_roll - roll) - self.kd_att* ang_vel[0]
        pitch_torque= self.kp_att* (des_pitch - pitch) - self.kd_att* ang_vel[1]

        yaw_torque= self.kp_yaw*(0- yaw) - self.kd_yaw* ang_vel[2]
        thrust_delta= self.kp_pos*pos_err[2] - self.kd_pos* vel[2]

        # 4 output action

        return np.array([thrust_delta,roll_torque,pitch_torque,yaw_torque])

