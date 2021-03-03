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
            0.0, # gBx ... 10
            0.0, # gBy ... 11
            0.0, # gBz ... 12
            0.0, # aBx ... 13
            0.0, # aBy ... 14
            0.0, # aBz ... 15
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
            1.0, # gBx ... 10
            1.0, # gBy ... 11
            1.0, # gBz ... 12
            1.0, # aBx ... 13
            1.0, # aBy ... 14
            1.0, # aBz ... 15
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
            1.0, # gBx ... 10
            1.0, # gBy ... 11
            1.0, # gBz ... 12
            1.0, # aBx ... 13
            1.0, # aBy ... 14
            1.0, # aBz ... 15
        ]) * 0.001


        # Process Noise
        accelNoise = 0.01
        gyroNoise = 0.01
        gyroDriftNoise = 0.01
        accelDriftNoise = 0.01
        self.setProcessNoise(accelNoise, gyroNoise, gyroDriftNoise, accelDriftNoise)

        # Measurment noise (accel, mag)
        sigma_accel = 0.1
        sigma_mag   = 0.1
        self.setMeasurmentNoise(sigma_accel, sigma_mag)

        self.z   = np.zeros(6)
        self.hx  = np.zeros(6)
        self.res = np.zeros(6)
        self.S   = np.zeros((6,6))

        self.g_L = np.array([0,0,9.81])
        self.mag_L = np.array([1.0, 0, 0])
        self._I = np.eye(len(self.state))

    def setProcessNoise(self, accelNoise, gyroNoise, gyroDriftNoise, accelDriftNoise):
        self.accelNoise = accelNoise
        self.gyroNoise = gyroNoise
        self.gyroDriftNoise = gyroDriftNoise
        self.accelDriftNoise = accelDriftNoise

    def setMeasurmentNoise(self, accelNoise, magNoise):
        self.sigma_accel = accelNoise
        self.sigma_mag   = magNoise
        self.R = np.diag( np.hstack(( [self.sigma_accel]*3, [self.sigma_mag]*3 )) )


    def getState(self):
        return np.copy(self.state)

    def predict(self,u):

        # Extract States from State vector
        rL = self.state[0:3]
        vL = self.state[3:6]
        q_toLfromB = self.state[6:10]
        gyroBias = self.state[10:13]
        accelBias = self.state[13:16]

        # Extract Inputs from Input vector
        wMeas = u[0:3]
        aMeas = u[3:6]

        # Adjust for bias
        wB = wMeas #- gyroBias
        aB = aMeas #- accelBias

        # State transition Models
        # Pos L
        self.state[0:3] = rL + vL*self.dt

        # Vel L
        aL = qRot(q_toLfromB, aB) + self.g_L
        self.state[3:6] = vL + aL*self.dt

        # q_toLfromB
        expq_wt = expq(wB*self.dt)
        self.state[6:10] = skew4L(q_toLfromB).dot( expq_wt )

        # Gyro Bias
        self.state[10:13] = gyroBias

        # Accel Bias
        self.state[13:16] = accelBias

        # State Transition Jacobian
        qw = np.array(q_toLfromB[0])
        qv = np.array(q_toLfromB[1:])
        QF = np.hstack((qv[:,np.newaxis],np.eye(3)))
        dvdq = np.vstack((
            np.hstack((0,aB)),
            np.hstack((aB[:,np.newaxis], -skew3(qv)))
        ))
        dvdq = 2*QF.dot(dvdq)
        C_toLfromB = quat2dcm(q_toLfromB)
        box = np.vstack((
            -qv,
            qw*np.eye(3) + skew3(qv)
        ))

        # Normalize quaternion
        self.normalizeQuaternion()

        F = np.zeros((16,16))
        F[0:3,3:6] = np.eye(3)
        F[3:6,6:10] = dvdq
        F[3:6,13:16] = -C_toLfromB
        F[6:10,6:10] = 0.5*skew4R(wB)
        F[6:10,10:13] = -0.5*box
        F = F*self.dt + np.eye(16)

        # Process Noise Matrix
        dt2 = self.dt**2
        dt4 = self.dt**4

        Qr = self.accelNoise**2 * dt4 * np.eye(3)
        Qv = self.accelNoise**2 * dt2 * np.eye(3)

        qw = q_toLfromB[0]
        qx = q_toLfromB[1]
        qy = q_toLfromB[2]
        qz = q_toLfromB[3]
        Qq = np.array([
            [1-qw*qw,  -qx*qw,  -qy*qw,  -qz*qw],
            [ -qw*qx, 1-qx*qx,  -qy*qx,  -qz*qx],
            [ -qw*qy,  -qx*qy, 1-qy*qy,  -qz*qy],
            [ -qw*qz,  -qx*qz,  -qy*qz, 1-qz*qz]
        ])
        Qq *= self.gyroNoise**2 * dt2/4


        Qgd = self.gyroDriftNoise**2 * dt2
        Qad = self.accelDriftNoise**2 * dt2

        Q = np.zeros((16,16))
        Q[0:3,0:3] = Qr
        Q[3:6,3:6] = Qv
        Q[6:10,6:10] = Qq
        Q[10:13,10:13] = Qgd * 0
        Q[13:16,13:16] = Qad

        # Update Prediction Covariance
        self.P = F.dot(self.P).dot(F.T) + Q

        # Save
        self.state_prior = np.copy(self.state)
        self.P_prior = np.copy(self.P)
        self.F = np.copy(F)
        self.Q = np.copy(Q)
        self.wB = np.copy(wB)
        self.aB = np.copy(aB)
        self.aL = np.copy(aL)
        self.aMeas = np.copy(aMeas)
        self.wMeas = np.copy(wMeas)


    def update(self,z):



        # Extract States from State vector
        rL = self.state[0:3]
        vL = self.state[3:6]
        q_toLfromB = self.state[6:10]
        gyroBias = self.state[10:13]
        accelBias = self.state[13:16]

        H = self.dhdx()

        PHT = self.P.dot(H.T)
        self.S = H.dot(PHT) + self.R
        #print(self.S.diagonal())
        try:
            self.K = PHT.dot( np.linalg.inv(self.S) )
        except:
            breakpoint()

        self.hx = self.h()
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



    def h(self):
        q_toLfromB = self.state[6:10] # q_toLfromB

        # Neglicting linear acceleration
        hx_accel = -qRot(qConj(q_toLfromB), self.g_L)
        hx_mag   =  qRot(qConj(q_toLfromB), self.mag_L)

        hx = np.hstack( (hx_accel, hx_mag) )
        return hx

    def dhdx(self):
        g = self.g_L[2]
        bx,by,bz = self.mag_L
        qw,qx,qy,qz = self.state[6:10]

        dhdx = np.array([
            [0, 0, 0, 0, 0, 0,    2*g*qy,  -2*g*qz,   2*g*qw,  -2*g*qx, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,   -2*g*qx,  -2*g*qw,  -2*g*qz,  -2*g*qy, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,   -2*g*qw,   2*g*qx,   2*g*qy,  -2*g*qz, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,   2*bx*qw,  2*bx*qx, -2*bx*qy, -2*bx*qz, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,  -2*bx*qz,  2*bx*qy,  2*bx*qx, -2*bx*qw, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0,   2*bx*qy,  2*bx*qz,  2*bx*qw,  2*bx*qx, 0, 0, 0, 0, 0, 0],
        ])

        return dhdx

    def get_q_toLfromB(self):
        return self.state[6:10]




