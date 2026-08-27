from os import access

import numpy as np
from scipy.spatial.transform import Rotation

class PIDHoverController:
    def __init__(self, kp_pos, kd_pos, kp_att, kd_att, kp_yaw, kd_yaw, max_tilt_rad=0.3,
                 ki_pos=0.0, ki_att=0.0, ki_yaw=0.0,
                 integral_limit_pos=0.3, integral_limit_att=0.3):

        #p - proportional gain, how hard to react to current error (how far rn)
        #d - derivative gain, how hard to react to rate of change (prevents overshoot)
        #i - integral gain, how hard to react to accumulated error (kills steady-state offset)

        # pos control kp pushes towards the target kd damps based on velocity , output is desired acceleration
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.ki_pos = ki_pos

        # attitude kp pushes toward desired roll pitch the current roll pitch kd damps based on the acceleration
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.ki_att = ki_att

        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.ki_yaw = ki_yaw
        self.max_tilt_rad = max_tilt_rad

        self.integral_limit_pos = integral_limit_pos
        self.integral_limit_att = integral_limit_att

        self.integral_pos = np.zeros(3)
        self.integral_att = np.zeros(2)  # roll, pitch
        self.integral_yaw = 0.0

    def reset(self):
        self.integral_pos = np.zeros(3)
        self.integral_att = np.zeros(2)
        self.integral_yaw = 0.0

    def compute_action(self, drone_state, target_pos, dt=1 / 240):

        # 1 read state
        pos= np.array([drone_state.position.x, drone_state.position.y, drone_state.position.z])
        vel=np.array([drone_state.velocity.x, drone_state.velocity.y, drone_state.velocity.z])
        ang_vel=np.array([drone_state.angular_velocity.x, drone_state.angular_velocity.y, drone_state.angular_velocity.z])
        roll, pitch, yaw = Rotation.from_quat([
            drone_state.orientation.x, drone_state.orientation.y,
            drone_state.orientation.z, drone_state.orientation.w
        ]).as_euler("xyz")

        pos_err= target_pos - pos

        self.integral_pos = np.clip(self.integral_pos + pos_err * dt,
                                     -self.integral_limit_pos, self.integral_limit_pos)

        # 2 outer loop
        accel_cmd= self.kp_pos* pos_err - self.kd_pos* vel + self.ki_pos* self.integral_pos

        des_roll=np.clip(-accel_cmd[1]/9.81,-self.max_tilt_rad,self.max_tilt_rad)
        des_pitch=np.clip(accel_cmd[0]/9.81,-self.max_tilt_rad,self.max_tilt_rad)

        att_err = np.array([des_roll - roll, des_pitch - pitch])
        self.integral_att = np.clip(self.integral_att + att_err * dt,
                                     -self.integral_limit_att, self.integral_limit_att)

        self.integral_yaw = np.clip(self.integral_yaw + (0 - yaw) * dt,
                                     -self.integral_limit_att, self.integral_limit_att)

        # 3 inner loop
        roll_torque= self.kp_att* att_err[0] - self.kd_att* ang_vel[0] + self.ki_att* self.integral_att[0]
        pitch_torque= self.kp_att* att_err[1] - self.kd_att* ang_vel[1] + self.ki_att* self.integral_att[1]

        yaw_torque= self.kp_yaw*(0- yaw) - self.kd_yaw* ang_vel[2] + self.ki_yaw* self.integral_yaw
        thrust_delta= self.kp_pos*pos_err[2] - self.kd_pos* vel[2] + self.ki_pos* self.integral_pos[2]

        # 4 output action

        return np.array([thrust_delta,roll_torque,pitch_torque,yaw_torque])

