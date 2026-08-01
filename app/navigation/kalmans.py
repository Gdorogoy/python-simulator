import numpy as np


"""
x_hat | state estimate |vector of 6 | current best guess: position and velocity
P | state covariance | vector of 6 | how uncertain each part of that guess is
F | state transition model | matrix of 6x6 | encodes the motion model, no measurement involved
B | control input model | (0 because of u) | maps a known control input into the state
u | control vector | (0 in the RL case because the accelerations is unknow) | the known control input itself, if any
q | process model | scalar  | how much you distrust your own motion model 
Q | process model | matrix of 6x6 | how much you distrust your own motion model in matrix form
H | measurements model | matrix of 6x3 | maps state into what the sensor actually reads
r | measurements noise variance | scalar | how noisy the sensors is
R | measurements noise variance  | matrix of 6x6 | how noisy the sensors is in matrix form
K | kalman gain | scalar | how much to trust new measurement vs prediction
z | measurements vector | vector of 3 only [x,y,z] | the raw noisy sensor reading this cycle
I | identity matrix | matrix pf 6x6 | does nothing when multiplied



inner methods:

R=I * r

Q=I * q


prediction methods:

x_hat  { n|n-1 }  = F*x_hat { n-1|n-1 }+ B*u { n}

P { n|n-1 } = F*P { n-1|n-1} * F transpose + Q

update methods::

K {n}= P { n|n-1} * H transpose *(H * P {n| n-1 } H transpose + R) ^ -1

x_hat { n|n} = x_hat {n|n-1} + K* (z {n} - H * x_hat { n| n-1} )

P { n|n }= (I - K { n} * H )* P { n|n-1 }

"""


class Kalman:

    def __init__(self,position,velocity,q,r,dt):
        self.x_hat= np.array([position[0],position[1],position[2] , velocity[0],velocity[1],velocity[2]])

        self.Q=np.identity(6)* q
        self.R=np.identity(3)* r

        self.P = np.identity(6)

        self.H=np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )

        self.dt= dt
        self.F= np.array([
            [1, 0, 0, dt, 0, 0], #px+vx
            [0, 1, 0, 0, dt, 0], #py+vy
            [0, 0, 1, 0, 0, dt], #pz+vz
            [0, 0, 0, 1, 0, 0],   #vx
            [0, 0, 0, 0, 1, 0],  #vy
            [0, 0, 0, 0, 0, 1], #vz
            ]
        )

        print("#-----KALMAN----OUTPUT-----#")
        self.display("x_hat",self.x_hat)
        self.display("F",self.F)
        self.display("Q",self.Q)
        self.display("R",self.R)
        self.display("P",self.P)
        self.display("H",self.H)





    def estimate_state(self,F,x_hat_prev):
        est_x_hat= F @ x_hat_prev
        return est_x_hat


    def estimated_state_covariance(self,F, P_prev , Q):
        P_est = F@ P_prev @ np.linalg.matrix_transpose(F) + Q
        return P_est


    def kalman_gain(self,P_est,H,R):
        K= P_est @ np.linalg.matrix_transpose(H) @ np.linalg.inv( (H @ P_est @ np.linalg.matrix_transpose(H) + R ))
        return K

    def update_estimated_state(self,est_x_hat,K,z,H):
        curr_x_hat= est_x_hat + K @ (z - H @ est_x_hat)
        return curr_x_hat

    def update_estimated_state_covariance(self,K,H,P_est):
        P_curr=(np.identity(6,float)-K@H)@ P_est
        return P_curr


    def update(self,z):
        K=self.kalman_gain(self.P_est,self.H,self.R)

        self.display("K",K)

        curr_x_hat=self.update_estimated_state(self.est_x_hat,K,z,self.H)

        self.display("predicte_x_hat",curr_x_hat)

        P_curr=self.update_estimated_state_covariance(K,self.H,self.P_est)

        self.display("predicted P",P_curr)

        self.x_hat=curr_x_hat
        self.P=P_curr





    def predict(self):
        est_x_hat=self.estimate_state(self.F,self.x_hat)

        self.display("x_hat",est_x_hat)
        self.display("F",self.F)
        self.display("Q",self.Q)

        P_est=self.estimated_state_covariance(self.F,self.P,self.Q)

        self.display("P_est",P_est)

        self.P_est=P_est
        self.est_x_hat=est_x_hat

    def loop(self,measurements):
        for z in measurements:
            self.predict()
            self.update(z)


        print(f"final predicted pos")
        print(f"{self.x_hat}")


    def display(self,name,val):
        print(f"              {name}")
        print("============================================")
        print(val)
        print("============================================")


def test():
    true_pos = np.array([0., 0., 0.])
    true_vel = np.array([0., 1., 2.])
    dt = 0.1
    filter = Kalman(true_pos.tolist(), true_vel.tolist(), q=0.12, r=0.34, dt=dt)

    for _ in range(20):
        true_pos = true_pos + true_vel * dt  # ground truth advances
        z = true_pos + np.random.normal(0, 0.34 ** 0.5, 3)  # noisy sensor
        filter.predict()
        filter.update(z)

    print(f"true pos: {true_pos} | true vel: {true_vel} ")
    print(f"estimated  {filter.x_hat}")
    combined=np.array([true_pos[0],true_pos[1],true_pos[2],true_vel[0],true_vel[1],true_vel[2]])

    rel,acc,abs=calc_diff(combined,filter.x_hat)

    print(f"relative error: {rel}")
    print(f"absolute error: {abs}")
    print(f"         acc  : {acc}")





def calc_diff(expected, actual):
    diff_abs = np.zeros(6)
    diff_relative = np.zeros(6)
    diff_accuracy = np.zeros(6)

    for i in range(6):
        diff_abs[i] = abs(expected[i] - actual[i])
        if abs(expected[i]) > 1e-8:
            diff_relative[i] = diff_abs[i] / abs(expected[i])
            diff_accuracy[i] = actual[i] / expected[i]
        else:
            diff_relative[i] = float('nan')
            diff_accuracy[i] = float('nan')

    return diff_relative, diff_accuracy, diff_abs


if __name__=="__main__":
    test()
