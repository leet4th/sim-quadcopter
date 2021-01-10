#!/usr/bin/env python3

import numpy as np
from numpy.random import randn
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from copy import deepcopy
import math
from math import radians, sin, cos, acos
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise

from IPython import embed as breakpoint

from transform import *

RAD2DEG = 180.0/np.pi
DEG2RAD = 1/RAD2DEG


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




class Model6DOF():
    def __init__(self, dt):

        # Time
        tStart = 0.0
        self.dt = dt

        # Model Constants
        self.m   = 100.0
        self.g   = 9.81
        self.Ixx = 10000.0
        self.Iyy = 10000.0
        self.Izz = 10000.0
        self.J = np.diag([self.Ixx,self.Iyy,self.Izz])
        self.Jinv = np.linalg.inv(self.J)

        self.g_L = np.array([0,0,self.g])
        self.mag_L = np.array([1.0, 0, 0])

        # Measurement Model Parameters
        self.gyroBias   = np.array([0,0,0])
        self.gyroNoise  = np.array([1.,1.,1.]) * (1.0e-1)**2
        self.accelBias  = np.array([0,0,0])
        self.accelNoise = np.array([1.,1.,1.]) * (1.0e-1)**2
        self.magBias    = np.array([0,0,0])
        self.magNoise   = np.array([1.,1.,1.]) * (1.0e-1)**2

        # state vector
        self.state = np.zeros(13)
        self.state[0]  = 0.0 # x
        self.state[1]  = 0.0 # y
        self.state[2]  = 0.0 # z
        self.state[3]  = 0.0 # vx
        self.state[4]  = 0.0 # vy
        self.state[5]  = 0.0 # vz
        self.state[6]  = 1.0 # qw
        self.state[7]  = 0.0 # qx
        self.state[8]  = 0.0 # qy
        self.state[9]  = 0.0 # qz
        self.state[10] = 0.0 # wx
        self.state[11] = 0.0 # wy
        self.state[12] = 0.0 # wz

        # dstate vector
        self.dstate = np.zeros_like(self.state)

        # Perform secondary calcs using current state and dstate vector
        self.calc()

        # Integrator
        self.integrator = ode(self.dstate_calc)

    def getState(self):
        return np.copy(self.state)

    def getDState(self):
        return np.copy(self.dstate)

    def unpackState(self):
        pos      = self.state[0:3]
        vel      = self.state[3:6]
        accel    = self.dstate[3:6]
        q        = self.state[6:10]
        w        = self.state[10:13]
        angAccel = self.dstate[10:13]

        return pos,vel,accel,q,w,angAccel

    def dstate_calc(self, t, state, cmd):

        self.normalizeQuaternion()

        r_BwrtLexpL = self.state[0:3]
        v_BwrtLexpL = self.state[3:6]
        q_toLfromB = self.state[6:10]
        wb = self.state[10:13]

        qmag = np.sqrt(q_toLfromB.dot(q_toLfromB))
        q_toLfromB = q_toLfromB / qmag

        Fx,Fy,Fz,Mx,My,Mz = cmd
        Fb = np.array([Fx,Fy,Fz])
        Mb = np.array([Mx,My,Mz])

        m   = self.m
        g_expL = self.g_L
        g   = self.g
        Ixx = self.Ixx
        Iyy = self.Iyy
        Izz = self.Izz
        J = self.J
        Jinv = self.Jinv

        #a_BwrtLexpB = 1/m*Fb + self.qRot(self.quatConj(q_toLfromB),g_L) - self.skew3(wb).dot(v_BwrtLexpB)
        #wbDot = Jinv.dot( -self.skew3(wb).dot(J).dot(wb) + Mb)
        #qDot = 0.5*self.skew4Left(q_toLfromB)[:,1:].dot(wb)
        ##qDot += 2.0 * (1.0 -q_toLfromB.dot(q_toLfromB)) * q_toLfromB

        a_BwrtLexpL = 1/m*qRot(q_toLfromB,Fb) + g_expL
        wbDot = Jinv.dot( -skew3(wb).dot(J).dot(wb) + Mb)
        qDot = 0.5*skew4L(q_toLfromB)[:,1:].dot(wb)

        self.dstate = np.hstack((v_BwrtLexpL, a_BwrtLexpL, qDot, wbDot))

        return self.dstate

    def update(self,t,dt,cmd):
        # Integrate
        #self.state = self.integrator.integrate(t,t+dt,step=True)
        self.integrator.set_integrator('dopri5',
                atol=1e90, rtol=1e90, nsteps=1,
                first_step=self.dt, max_step=self.dt)
        self.integrator.set_initial_value(self.state)
        self.integrator.set_f_params(cmd)
        self.state = self.integrator.integrate(dt)
        returnCode = self.integrator.get_return_code()
        if returnCode == 1:
            #print('Integration successful.')
            pass
        if returnCode == 2:
            print('Integration successful (interrupted by solout).')
        if returnCode == -1:
            print('Input is not consistent.')
        if returnCode == -2:
            print('Larger nsteps is needed.')
        if returnCode == -3:
            print('Step size becomes too small.')
        if returnCode == -4:
            print('Problem is probably stiff (interrupted).')

        # Post integration step operations
        self.normalizeQuaternion()
        self.calc()

    def calc(self):
        state  = self.getState()
        dstate = self.getDState()

        self.r_BwrtLexpL = state[0:3]
        self.v_BwrtLexpL = state[3:6]
        self.q_toLfromB  = state[6:10]
        self.wb          = state[10:13]
        self.a_BwrtLexpL = dstate[3:6]
        self.wbDot       = dstate[10:13]

        self.q_toBfromL  = qConj(self.q_toLfromB)
        self.r_BwrtLexpB = qRot( self.q_toBfromL, self.r_BwrtLexpL )
        self.v_BwrtLexpB = self.v_BwrtLexpL - skew3(self.wb).dot(self.r_BwrtLexpB)
        #self.a_BwrtLexpB = self.a_BwrtLexpL - skew3(self.wb).dot(self.v_BwrtLexpB)

        self.a_BwrtLexpB = self.a_BwrtLexpL - skew3( self.wbDot ).dot( self.v_BwrtLexpB )

        self.v_BwrtLexpB2 = qRot( self.q_toBfromL, self.v_BwrtLexpL )
        self.a_BwrtLexpB2 = qRot( self.q_toBfromL, self.a_BwrtLexpL )

        self.euler321    = quat2euler321( self.q_toBfromL )
        self.g_B         = qRot( self.q_toBfromL, self.g_L )

        self.wMeas = self.gyroMeasurementModel()
        self.aMeas = self.accelMeasurementModel()
        self.mMeas = self.magMeasurementModel()


    def gyroMeasurementModel(self):
        return self.wb + self.gyroBias + self.gyroNoise*randn(3)

    def accelMeasurementModel(self):
        spForce_B  = self.a_BwrtLexpB  - self.g_B
        spForce_B2 = self.a_BwrtLexpB2 - self.g_B
        #print(f'{self.a_BwrtLexpB} {self.a_BwrtLexpB2} {self.g_B}')
        #print(f'{spForce_B}    {spForce_B2}')
        spForce_B  = -self.g_B
        return spForce_B + self.accelBias + self.accelNoise*randn(3)

    def magMeasurementModel(self):
        mag_B = qRot(self.q_toBfromL, self.mag_L)
        return mag_B + self.magBias + self.magNoise*randn(3)

    def normalizeQuaternion(self,t=None,y=None):
        q = self.state[6:10]
        qmagsq = q.dot(q)
        if np.abs(1.0 - qmagsq) < 2.107342e-08:
            # First order pade approximation
            q = q * 2.0/(1.0 + qmagsq)
        else:
            q = q / np.sqrt(qmagsq)
        self.state[6:10] = q

class ModelOutput():
    def __init__(self,varList,model):
        self.model = model
        self.data = {}
        for var in varList:
            self.data[var] = []

    def append(self):
        for var in self.data.keys():
            self.data[var].append( np.copy(getattr(self.model,var)) )

    def process(self):
        for var in self.data.keys():
            self.data[var] = np.asarray(self.data[var]).T




dt = .001 # sec
tStart = 0.0
tEnd = 10.0

nSteps = int((tEnd-tStart)/dt) + 1
time,dt = np.linspace(tStart, tEnd, num=nSteps, retstep=True)

model = Model6DOF(dt)
ekf = AttitudeEKF(dt)

# Noise
model.gyroNoise  *= 1
model.accelNoise *= 1
model.magNoise   *= 1

#qStart = euler3212quat(np.array([0,90,0])*DEG2RAD)
#model.state[6:10] = qStart


# Fbx,Fby,Fbz,Mbx,Mby,Mbz
cmd = np.array([
    0,
    0,
    0,
    1000,
    1000,
    1000,
])


state = []
dstate = []


modelVarList = [
    'r_BwrtLexpB',
    'v_BwrtLexpB',
    'q_toLfromB',
    'wb',
    'a_BwrtLexpB',
    'wbDot',
    'q_toBfromL',
    'r_BwrtLexpL',
    'v_BwrtLexpL',
    'a_BwrtLexpL',
    'euler321',
    'wMeas',
    'aMeas',
    'mMeas',
]
modelData = ModelOutput(modelVarList, model)

ekfVarList = [
    'state',
    'z',
    'hx',
    'res',
    'P',
    'S',
]
ekfData = ModelOutput(ekfVarList,ekf)

lastProgress = -1
showProgress = True
showPlots = True
wantAnimation = True
for tk in time:


    progress = int(tk/tEnd*100)
    if progress % 10 == 0 and lastProgress != progress and showProgress:
        lastProgress = progress
        print(f'\t{progress}%')

    # True
    wb = model.wb
    # Measurements
    wMeas = model.wMeas
    aMeas = model.aMeas
    mMeas = model.mMeas

    # EKF
    zMeas = np.hstack( (aMeas, mMeas) )
    ekf.predict(wMeas)
    #ekf.state = model.q_toLfromB + 0.05*randn()
    ekf.update(zMeas)

    # Cmd
    if wb[0]*RAD2DEG >= 5:
        cmd[3] = 0
    #cmd[3] = 1000.0*np.sin(45*DEG2RAD*tk)

    if wb[1]*RAD2DEG >= 15:
        cmd[4] = 0

    if wb[2]*RAD2DEG >= 2:
        cmd[5] = 0

    #if tk >= 1.0:
    #    cmd[0:3] = 0.0

    # Update sim models
    model.update(tk,dt,cmd)

    # Store data
    modelData.append()
    ekfData.append()


# Post processing
time = np.asarray(time).T
modelData.process()
ekfData.process()


# Plots
if showPlots:
    fig,ax = plt.subplots(1,1,sharex=True)
    for k in range(4):
        ax.plot(time,ekfData.data['state'][k,:],'r.')
        ax.plot(time,modelData.data['q_toLfromB'][k,:],'b')
    ax.grid()
    ax.set_ylim([-1.1,1.1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q_toLfromB')


    fig,ax = plt.subplots(1,1,sharex=True)
    for k in range(4):
        res = modelData.data['q_toLfromB'][k,:] - ekfData.data['state'][k,:]
        ax.plot(time,res,'r')
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('res (q_toLfromB)')




    fig,ax = plt.subplots(1,1,sharex=True)
    for k in range(4):
        ax.plot(time,ekfData.data['P'][k,k,:])
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('P')

    fig,ax = plt.subplots(1,1,sharex=True)
    ax.plot(time,modelData.data['wMeas'][0,:]*RAD2DEG,'r.',label='wMeas x')
    ax.plot(time,modelData.data['wMeas'][1,:]*RAD2DEG,'g.',label='wMeas y')
    ax.plot(time,modelData.data['wMeas'][2,:]*RAD2DEG,'b.',label='wMeas z')
    ax.plot(time,modelData.data['wb'][0,:]*RAD2DEG,'r-',label='wx')
    ax.plot(time,modelData.data['wb'][1,:]*RAD2DEG,'g-',label='wy')
    ax.plot(time,modelData.data['wb'][2,:]*RAD2DEG,'b-',label='wz')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('wb (deg/s)')

    #fix,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(time,modelData.data['euler321'][0,:]*RAD2DEG,label='yaw')
    #ax.plot(time,modelData.data['euler321'][1,:]*RAD2DEG,label='pitch')
    #ax.plot(time,modelData.data['euler321'][2,:]*RAD2DEG,label='roll')
    #ax.legend()
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('euler321 (deg)')

    #fig,ax = plt.subplots(3,2,sharex=True)
    #ax[0,0].plot(time,ekfData.data['hx'][0,:],'r.', label='hx ax')
    #ax[1,0].plot(time,ekfData.data['hx'][1,:],'r.', label='hx ay')
    #ax[2,0].plot(time,ekfData.data['hx'][2,:],'r.', label='hx az')
    #ax[0,1].plot(time,ekfData.data['hx'][3,:],'r.', label='hx mx')
    #ax[1,1].plot(time,ekfData.data['hx'][4,:],'r.', label='hx my')
    #ax[2,1].plot(time,ekfData.data['hx'][5,:],'r.', label='hx az')
    #ax[0,0].plot(time,ekfData.data['z'][0,:], 'b-', label='z ax')
    #ax[1,0].plot(time,ekfData.data['z'][1,:], 'b-', label='z ay')
    #ax[2,0].plot(time,ekfData.data['z'][2,:], 'b-', label='z az')
    #ax[0,1].plot(time,ekfData.data['z'][3,:], 'b-', label='z mz')
    #ax[1,1].plot(time,ekfData.data['z'][4,:], 'b-', label='z mz')
    #ax[2,1].plot(time,ekfData.data['z'][5,:], 'b-', label='z mz')
    #for k in range(3):
    #    ax[k,0].grid()
    #    ax[k,1].grid()
    #    ax[k,0].legend()
    #    ax[k,1].legend()
    #    ax[k,1].set_ylim([-1,1])
    #ax[2,0].set_xlabel('Time (s)')
    #ax[2,1].set_xlabel('Time (s)')
    #ax[0,0].set_title('Accel')
    #ax[0,1].set_title('Mag')

    fig,ax = plt.subplots(3,2,sharex=True)
    ax[0,0].fill_between(time,-ekfData.data['S'][0,0,:],ekfData.data['S'][0,0,:],facecolor='#ffff00',alpha=0.3)
    ax[1,0].fill_between(time,-ekfData.data['S'][1,1,:],ekfData.data['S'][1,1,:],facecolor='#ffff00',alpha=0.3)
    ax[2,0].fill_between(time,-ekfData.data['S'][2,2,:],ekfData.data['S'][2,2,:],facecolor='#ffff00',alpha=0.3)
    ax[0,1].fill_between(time,-ekfData.data['S'][3,3,:],ekfData.data['S'][3,3,:],facecolor='#ffff00',alpha=0.3)
    ax[1,1].fill_between(time,-ekfData.data['S'][4,4,:],ekfData.data['S'][4,4,:],facecolor='#ffff00',alpha=0.3)
    ax[2,1].fill_between(time,-ekfData.data['S'][5,5,:],ekfData.data['S'][5,5,:],facecolor='#ffff00',alpha=0.3)
    ax[0,0].plot(time,ekfData.data['S'][0,0,:], 'k:', label='S ax')
    ax[1,0].plot(time,ekfData.data['S'][1,1,:], 'k:', label='S ay')
    ax[2,0].plot(time,ekfData.data['S'][2,2,:], 'k:', label='S az')
    ax[0,1].plot(time,ekfData.data['S'][3,3,:], 'k:', label='S mx')
    ax[1,1].plot(time,ekfData.data['S'][4,4,:], 'k:', label='S my')
    ax[2,1].plot(time,ekfData.data['S'][5,5,:], 'k:', label='S mz')
    ax[0,0].plot(time,-ekfData.data['S'][0,0,:],'k:', label='S ax')
    ax[1,0].plot(time,-ekfData.data['S'][1,1,:],'k:', label='S ay')
    ax[2,0].plot(time,-ekfData.data['S'][2,2,:],'k:', label='S az')
    ax[0,1].plot(time,-ekfData.data['S'][3,3,:],'k:', label='S mx')
    ax[1,1].plot(time,-ekfData.data['S'][4,4,:],'k:', label='S my')
    ax[2,1].plot(time,-ekfData.data['S'][5,5,:],'k:', label='S mz')
    ax[0,0].plot(time,ekfData.data['res'][0,:], 'r.', label='res ax')
    ax[1,0].plot(time,ekfData.data['res'][1,:], 'r.', label='res ay')
    ax[2,0].plot(time,ekfData.data['res'][2,:], 'r.', label='res az')
    ax[0,1].plot(time,ekfData.data['res'][3,:], 'r.', label='res mx')
    ax[1,1].plot(time,ekfData.data['res'][4,:], 'r.', label='res my')
    ax[2,1].plot(time,ekfData.data['res'][5,:], 'r.', label='res az')
    for k in range(3):
        ax[k,0].grid()
        ax[k,1].grid()
        ax[k,0].legend()
        ax[k,1].legend()
        ax[k,1].set_ylim([-1,1])
    ax[2,0].set_xlabel('Time (s)')
    ax[2,1].set_xlabel('Time (s)')
    ax[0,0].set_title('Accel')
    ax[0,1].set_title('Mag')


    #fig,ax = plt.subplots(1,1,sharex=True)
    #for k in range(3):
    #    ax.plot(time,ekfData.data['res'][k,:])
    #ax.grid()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('ekf res')


    #fix,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(time,modelData.data['r_BwrtLexpL'][0,:],label='x')
    #ax.plot(time,modelData.data['r_BwrtLexpL'][1,:],label='y')
    #ax.plot(time,modelData.data['r_BwrtLexpL'][2,:],label='z')
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('r_BwrtLexpL')

    #fix,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(time,modelData.data['a_BwrtLexpB'][0,:],label='x')
    #ax.plot(time,modelData.data['a_BwrtLexpB'][1,:],label='y')
    #ax.plot(time,modelData.data['a_BwrtLexpB'][2,:],label='z')
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('a_BwrtLexpB')

    #fix,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(time,modelData.data['a_BwrtLexpL'][0,:],label='x')
    #ax.plot(time,modelData.data['a_BwrtLexpL'][1,:],label='y')
    #ax.plot(time,modelData.data['a_BwrtLexpL'][2,:],label='z')
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('a_BwrtLexpL')


    #fix,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(modelData.data['r_BwrtLexpL'][1,:],modelData.data['r_BwrtLexpL'][0,:],label='model')
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('East (y)')
    #ax.set_ylabel('North (x)')
    #ax.set_title('r_BwrtLexpL')


    #fix,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(modelData.data['r_BwrtLexpB'][1,:],modelData.data['r_BwrtLexpB'][0,:],label='model')
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('Body-Y')
    #ax.set_ylabel('Body-X')
    #ax.set_title('r_BwrtLexpB')

    plt.show()

if wantAnimation:

    import pygame
    from attitude3d import OpenLoopAttitude, initializeCube, ProjectionViewer
    sys = OpenLoopAttitude()
    block = initializeCube(3,2,0.2,sys)
    pv = ProjectionViewer(640, 480, block)

    while(True):
        #for q_toBfromL in modelData.data['q_toBfromL'][:,::100].T:
        for q_toBfromL in ekfData.data['state'][:,::100].T:
            pv.clock.tick(int(model.dt*50*1000))

            # sys update attitude
            block.sys.setQuat(q_toBfromL)

            # display updated attitude
            pv.display()
            pygame.display.flip()



