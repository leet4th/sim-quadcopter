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
            1.0, # qw
            0.0, # qx
            0.0, # qy
            0.0, # qz
        ])

        # State covariance matrix
        self.P = np.diag([
            1.0, # qw
            1.0, # qx
            1.0, # qy
            1.0, # qz
        ]) * 0.001

        # Process noise (gyro?)
        self.Q = np.eye(4) * 0.000001

        # Measurment noise (accel, mag)
        self.sigma_accel = 0.1
        self.sigma_mag   = 0.1
        self.R = np.diag( np.hstack(( [self.sigma_accel]*3, [self.sigma_mag]*3 )) )

        self.z   = np.zeros(6)
        self.hx  = np.zeros(6)
        self.res = np.zeros(6)
        self.S   = np.zeros((6,6))

        self.g_L = np.array([0,0,9.81])
        self.mag_L = np.array([1.0, 0, 0])
        self._I = np.eye(len(self.state))

    def getState(self):
        return np.copy(self.state)

    def predict(self,u):

        # Extract states
        q = self.state[0:4]
        gyroBias = np.array([0,0,0])

        w = u[:] - gyroBias

        expq_wt = expq(w*self.dt)

        # Next state vector
        qNext = skew4L(q).dot( expq_wt )

        # Pack state vector
        self.state[0:4] = qNext

        # Update Prediction Covariance
        # First order approx
        F = np.eye(4) + self.dt/2*skew4R(w)
        G = np.eye(4)

        self.P = F.dot(self.P).dot(F.T) + G.dot(self.Q).dot(G.T)

        # Save prior
        self.state_prior = np.copy(self.state)
        self.P_prior = np.copy(self.P)

    def update(self,z):

        # Extract states
        q = self.state[0:4]
        gyroBias = np.array([0,0,0])

        H = self.dhdx()

        PHT = self.P.dot(H.T)
        self.S = H.dot(PHT) + self.R
        self.K = PHT.dot( np.linalg.inv(self.S) )

        self.hx = self.h()
        self.res = z - self.hx

        self.state = self.state + self.K.dot(self.res)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        #I_KH = self._I - self.K.dot(H)
        #self.P = np.dot(I_KH,self.P).dot(I_KH.T) + np.dot(self.K, self.R).dot(self.K.T)
        #self.P = I_KH.dot(self.P)

        self.P = self.P - self.K.dot(self.S).dot(self.K.T)

        # Normalize quaternion
        q = self.state[0:4]
        qMag = np.sqrt(q.dot(q))
        J = q * q.reshape(-1,1) / qMag
        self.P = np.dot(J, self.P).dot(J.T)
        self.state[0:4] = q/qMag

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.state_post = self.state.copy()
        self.P_post = self.P.copy()


    def h(self):
        q = self.state[0:4] # q_toLfromB

        # Neglicting linear acceleration
        hx_accel = -qRot(qConj(q), self.g_L)
        hx_mag   =  qRot(qConj(q), self.mag_L)

        hx = np.hstack( (hx_accel, hx_mag) )
        return hx

    def dhdx(self):
        ax,ay,az = self.g_L
        bx,by,bz = self.mag_L
        qw,qx,qy,qz = self.state[0:4]

        dhdx_accel = np.array([
            [ 2*az*qy, -2*az*qz,  2*az*qw, -2*az*qx],
            [-2*az*qx, -2*az*qw, -2*az*qz, -2*az*qy],
            [-2*az*qw,  2*az*qx,  2*az*qy, -2*az*qz]
        ])

        dhdx_mag = np.array([
            [ 2*bx*qw, 2*bx*qx, -2*bx*qy, -2*bx*qz],
            [-2*bx*qz, 2*bx*qy,  2*bx*qx, -2*bx*qw],
            [ 2*bx*qy, 2*bx*qz,  2*bx*qw,  2*bx*qx]
        ])

        dh = np.vstack( (dhdx_accel, dhdx_mag) )
        return dh


