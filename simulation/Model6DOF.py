#!/usr/bin/env python3

import numpy as np
from numpy.random import randn
from scipy.integrate import ode
from copy import deepcopy
from math import radians, sin, cos, acos
import matplotlib.pyplot as plt

from transform import *

from IPython import embed as breakpoint

RAD2DEG = 180.0/np.pi
DEG2RAD = 1/RAD2DEG


class Model6DOF():
    def __init__(self, dt):

        # Time
        tStart = 0.0
        self.dt = dt

        # Model Constants
        self.mass= 1.2 # kg
        self.g   = 9.81
        self.Ixx = 0.0123
        self.Iyy = 0.0123
        self.Izz = 0.0123
        self.J = np.diag([self.Ixx,self.Iyy,self.Izz])
        self.Jinv = np.linalg.inv(self.J)

        self.g_L = np.array([0,0,self.g])
        self.mag_L = np.array([1.0, 0, 0])

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

    def calcFM(self,t,state,cmd):
        return cmd

    def calcMeasurementModel(self):
        pass

    def unpackState(self):
        pos      = self.state[0:3]
        vel      = self.state[3:6]
        accel    = self.dstate[3:6]
        q        = self.state[6:10]
        w        = self.state[10:13]
        angAccel = self.dstate[10:13]

        return pos,vel,accel,q,w,angAccel

    def dstate_calc(self, t, state, FM):

        Fb,Mb = FM

        self.normalizeQuaternion()

        r_BwrtLexpL = self.state[0:3]
        v_BwrtLexpL = self.state[3:6]
        q_toLfromB = self.state[6:10]
        wb = self.state[10:13]

        qmag = np.sqrt(q_toLfromB.dot(q_toLfromB))
        q_toLfromB = q_toLfromB / qmag


        m   = self.mass
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

        # Calc Body forces/moments from actuators
        Fx,Fy,Fz,Mx,My,Mz = self.calcFM(t,self.state,cmd)

        # Save Body Forces/Moments
        self.Fb = np.array([Fx,Fy,Fz])
        self.Mb = np.array([Mx,My,Mz])
        FM = (self.Fb, self.Mb)

        # Integrate
        self.integrator.set_integrator('dopri5',
                atol=1e90, rtol=1e90, nsteps=1,
                first_step=self.dt, max_step=self.dt)
        self.integrator.set_initial_value(self.state)
        self.integrator.set_f_params( FM )
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

        self.euler321_toBfromL    = quat2euler321( self.q_toBfromL )
        self.g_B         = qRot( self.q_toBfromL, self.g_L )

        self.euler321 = quat2euler321( self.q_toLfromB )

        self.calcMeasurementModel()


    def normalizeQuaternion(self,t=None,y=None):
        q = self.state[6:10]
        qmagsq = q.dot(q)
        if np.abs(1.0 - qmagsq) < 2.107342e-08:
            # First order pade approximation
            q = q * 2.0/(1.0 + qmagsq)
        else:
            q = q / np.sqrt(qmagsq)
        self.state[6:10] = q

class ModelQuadcopter(Model6DOF):
    def __init__(self, dt):

        # Motor Coefficients
        self.kF = np.ones(4)*1.076e-5 # N/(rad/s)^2
        self.kM = np.ones(4)*1.632e-7 # Nm/(rad/s)^2
        self.L = 0.16 # m

        # Measurement Model Parameters
        self.gyroBias   = np.array([0,0,0])
        self.gyroNoise  = np.array([1.,1.,1.]) * np.sqrt(30.0*DEG2RAD)/3
        self.accelBias  = np.array([0,0,0])
        self.accelNoise = np.array([1.,1.,1.]) * np.sqrt(0.4)/3
        self.magBias    = np.array([0,0,0])
        self.magNoise   = np.array([1.,1.,1.]) * np.sqrt(0.005)/3
        self.gpsBias    = np.array([0,0,0])
        self.gpsNoise   = np.array([1.,1.,1.]) * np.sqrt(1.8)/3

        # Model6DOF constructor must be called last due to inherited methods
        super().__init__(dt)

        self.hoverCmd = np.sqrt( self.mass*self.g/(4*self.kF) )

    def gyroMeasurementModel(self):
        return self.wb + self.gyroBias + self.gyroNoise*randn(3)

    def accelMeasurementModel(self):
        spForce_B  = self.a_BwrtLexpB  - self.g_B
        spForce_B2 = self.a_BwrtLexpB2 - self.g_B
        spForce_B  = -self.g_B
        return spForce_B + self.accelBias + self.accelNoise * randn(3)

    def magMeasurementModel(self):
        mag_B = qRot(self.q_toBfromL, self.mag_L)
        return mag_B + self.magBias + self.magNoise * randn(3)

    def gpsMeasurementModel(self):
        gpsPos_L = self.r_BwrtLexpL + self.gpsBias + self.gpsNoise * randn(3)
        return gpsPos_L

    def calcMeasurementModel(self):
        self.wMeas = self.gyroMeasurementModel()
        self.aMeas = self.accelMeasurementModel()
        self.mMeas = self.magMeasurementModel()
        self.rMeas = self.gpsMeasurementModel()

    def calcFM(self,t,state,cmd):
        # cmd = motor speeds
        # wm1,wm2,wm3,wm4 = cmd

        # Calc Motor force/moments
        self.Fm = self.kF*cmd*cmd
        self.Mm = self.kM*cmd*cmd

        Fm1,Fm2,Fm3,Fm4 = self.Fm
        Mm1,Mm2,Mm3,Mm4 = self.Mm

        # Calc Body Forces due to motors
        self.Fbm = np.array([
                        0,
                        0,
                       -(Fm1+Fm2+Fm3+Fm4)
                       ])

        # Calc Body Moments due to motors
        self.Mbm = np.array([
                        self.L*(Fm4-Fm2),
                        self.L*(Fm1-Fm3),
                        Mm2 + Mm4 - Mm1 -Mm3,
                       ])

        return np.hstack((self.Fbm,self.Mbm))


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


