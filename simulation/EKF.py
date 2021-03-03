#!/usr/bin/env python3

import numpy as np
from copy import deepcopy
from transform import *

from IPython import embed as breakpoint

class AttitudeEKF():
    def __init__(self,dt):
        self.dt = dt


        # Attitude only state vector
        self.state = np.array([
            0.0, # rLx ...  0
            0.0, # rLy ...  1
            0.0, # rLz ...  2
            0.0, # vLx ...  3
            0.0, # vLy ...  4
            0.0, # vLz ...  5
            1.0, # qw  ...  6
            0.0, # qx  ...  7
            0.0, # qy  ...  8
            0.0, # qz  ...  9
        ])

        # State covariance matrix
        self.P = np.diag([
            1.0, # rLx ...  0
            1.0, # rLy ...  1
            1.0, # rLz ...  2
            1.0, # vLx ...  3
            1.0, # vLy ...  4
            1.0, # vLz ...  5
            1.0, # qw  ...  6
            1.0, # qx  ...  7
            1.0, # qy  ...  8
            1.0, # qz  ...  9
        ]) * 0.001

        # Process noise
        self.Q = np.diag([
            1.0, # rLx ...  0
            1.0, # rLy ...  1
            1.0, # rLz ...  2
            1.0, # vLx ...  3
            1.0, # vLy ...  4
            1.0, # vLz ...  5
            1.0, # qw  ...  6
            1.0, # qx  ...  7
            1.0, # qy  ...  8
            1.0, # qz  ...  9
        ]) * 0.1


        # Process Noise
        std_accel = 0.1
        std_gyro  = 0.1
        self.setProcessNoise(std_accel, std_gyro)

        # Measurment noise (accel, mag)
        sigma_gps   = 0.05
        sigma_accel = 0.05
        sigma_mag   = 0.05
        self.setMeasurmentNoise(sigma_gps, sigma_accel, sigma_mag)

        self.z   = np.zeros(6)
        self.hx  = np.zeros(6)
        self.res = np.zeros(6)
        self.S   = np.zeros((6,6))

        self.g_L = np.array([0,0,9.81])
        self.mag_L = np.array([1.0, 0, 0])
        self._I = np.eye(len(self.state))


    def setProcessNoise(self, std_accel, std_gyro):
        self.sigSq_accel = std_accel**2
        self.sigSq_gyro = std_gyro**2
        self.calcProcessNoise()

    def setMeasurmentNoise(self, sigma_gps, sigma_accel, sigma_mag):
        self.sigSq_gps   = sigma_gps**2
        self.sigSq_accel = sigma_accel**2
        self.sigSq_mag   = sigma_mag**2
        self.R = np.diag( np.hstack((
            [self.sigSq_gps]*3,
            [self.sigSq_accel]*3,
            [self.sigSq_mag]*3
        )) )


    def getState(self):
        return np.copy(self.state)

    def calcStateModel(self, wB, aB):

        t = self.dt
        rL = self.state[0:3]
        vL = self.state[3:6]
        q_toLfromB = self.state[6:10]

        # Accel L
        aL = qRot(q_toLfromB, aB) + self.g_L

        # Pos L
        fx_r = rL + vL*t + 0.5*t**2*aL

        # Vel L
        fx_v = vL + aL*t

        # q_toLfromB
        expq_wt = expq(wB*self.dt)
        fx_q = skew4L(q_toLfromB).dot( expq_wt )

        fx = np.hstack((
            fx_r,
            fx_v,
            fx_q
            ))


        self.aL = np.copy(aL)

        return fx

    def calcStateModelJac(self, wB, aB, want_1stOrderApprox=False):

        t = self.dt
        rL = self.state[0:3]
        vL = self.state[3:6]
        q_toLfromB = self.state[6:10]

        if want_1stOrderApprox:
            dfdx_q = eye(4) + t/2*skew4Right(wB)
        else:
            dfdx_q = skew4R( expq( 0.5*t/2*wB ) )

        F = np.block([
            [      np.eye(3),     t*np.eye(3), 0.5*t**2*dRotdq(q_toLfromB, aB)],
            [      np.eye(3),     t*np.eye(3),        t*dRotdq(q_toLfromB, aB)],
            [np.zeros((4,3)), np.zeros((4,3)),                           dfdx_q],
        ])

        return F


    def calcProcessNoise(self):

        t = self.dt
        sigSq_accel = self.sigSq_accel
        sigSq_gyro  = self.sigSq_gyro
        qw,qx,qy,qz = self.state[6:10] # q_toLfromB

        # Q - Pos and Velocity
        Qpv = np.array([
            [t**4/4,      0,      0, t**3/2,      0,      0],
            [     0, t**4/4,      0,      0, t**3/2,      0],
            [     0,      0, t**4/4,      0,      0, t**3/2],
            [t**3/2,      0,      0,   t**2,      0,      0],
            [     0, t**3/2,      0,      0,   t**2,      0],
            [     0,      0, t**3/2,      0,      0,   t**2],
        ]) * sigSq_accel

        Qq = np.array([
            [1-qw**2,  -qx*qw,  -qy*qw,  -qz*qw],
            [ -qw*qx, 1-qx**2,  -qy*qx,  -qz*qx],
            [ -qw*qy,  -qx*qy, 1-qy**2,  -qz*qy],
            [ -qw*qz,  -qx*qz,  -qy*qz, 1-qz**2]
        ]) * t**2/4 * sigSq_gyro

        Q = np.block([
            [Qpv, np.zeros((6,4))],
            [np.zeros((4,6)), Qq],
        ])

        return Q


    def calcMeasurmentModel(self):
        rL = self.state[0:3]
        q_toLfromB = self.state[6:10] # q_toLfromB
        g_L = self.g_L
        mag_L = self.mag_L

        # Neglicting linear acceleration
        hx_gps   =  rL
        hx_accel = -qRot(qConj(q_toLfromB), g_L)
        hx_mag   =  qRot(qConj(q_toLfromB), mag_L)

        hx = np.hstack((
            hx_gps,
            hx_accel,
            hx_mag,
        ))

        return hx




    def calcMeasurmentModelJac(self):
        g_L = self.g_L
        mag_L = self.mag_L
        q_toLfromB = self.state[6:10]

        dhdx = np.block([
            [      np.eye(3), np.zeros((3,3)),         np.zeros((3,4))],
            [np.zeros((3,3)), np.zeros((3,3)), dVdq(q_toLfromB,  -g_L)],
            [np.zeros((3,3)), np.zeros((3,3)), dVdq(q_toLfromB, mag_L)],
        ])

        return dhdx


    def predict(self,u):

        # Extract Inputs from Input vector
        wMeas = u[0:3]
        aMeas = u[3:6]

        # Adjust for bias/filtering/calibration
        wB = wMeas
        aB = aMeas

        # State Transition
        self.state = self.calcStateModel(wB, aB)

        # State Transition Jacobian
        self.F = self.calcStateModelJac(wB, aB)

        # Process Noise Matrix
        self.Q = self.calcProcessNoise()

        # Update Prediction Covariance
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

        # Normalize quaternion
        self.normalizeQuaternion()

        # Save
        self.state_prior = np.copy(self.state)
        self.P_prior = np.copy(self.P)
        self.aMeas = np.copy(aMeas)
        self.wMeas = np.copy(wMeas)
        self.wB = np.copy(wB)
        self.aB = np.copy(aB)


    def update(self,z):

        H = self.calcMeasurmentModelJac()

        PHT = self.P.dot(H.T)
        self.S = H.dot(PHT) + self.R
        #print(self.S.diagonal())
        try:
            self.K = PHT.dot( np.linalg.inv(self.S) )
        except:
            breakpoint()

        self.hx = self.calcMeasurmentModel()
        self.res = z - self.hx

        self.state = self.state + self.K.dot(self.res)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - self.K.dot(H)
        self.P = np.dot(I_KH,self.P).dot(I_KH.T) + np.dot(self.K, self.R).dot(self.K.T)
        #self.P = I_KH.dot(self.P)

        #self.P = self.P - self.K.dot(self.S).dot(self.K.T)

        # Normalize quaternion
        self.normalizeQuaternion()

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.state_post = self.state.copy()
        self.P_post = self.P.copy()

    def normalizeQuaternion(self):
        # Normalize quaternion
        q = self.state[6:10]
        qMag = np.sqrt(q.dot(q))
        self.state[6:10] = q/qMag
        J = q * q.reshape(-1,1) / qMag
        self.P[6:10,6:10]= np.dot(J, self.P[6:10,6:10]).dot(J.T)

    def get_q_toLfromB(self):
        return self.state[6:10]

    def updateNav(self, u, z):

        # EKF Predict and Update loops
        self.predict(u)
        self.update(z)

        # Assemble data for control
        rL = self.state[0:3]
        vL = self.state[3:6]
        aL = self.aL
        q_toLfromB = self.state[6:10]
        wB = self.wB
        aB = self.aB

        navData = np.hstack((
            rL,
            vL,
            aL,
            q_toLfromB,
            wB,
            aB,
        ))

        #navData = {
        #        'rL':rL,
        #        'vL':vL,
        #        'aL':aL,
        #        'q_toLfromB':q_toLfromB,
        #        'wB':wB,
        #        'aB':aB,
        #}

        return navData





