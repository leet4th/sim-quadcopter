#!/usr/bin/env python3

import numpy as np
from math import radians, sin, cos, acos, asin
from transform import *

from IPython import embed as breakpoint

RAD2DEG = 180.0/np.pi
DEG2RAD = 1/RAD2DEG


class ControlLoop:

    def __init__(self, dt):
        self.dt = dt

        self.accel_cmd = np.array([0,0,1.])
        self.yaw_cmd = 0.0
        self.q_toBfromL = np.array([1.0,0,0,0])
        self.cmd = np.zeros(4)

    def readNavData(self, navData):
        self.rL = navData[0:3]
        self.vL = navData[3:6]
        self.aL = navData[6:9]
        self.q_toLfromB = navData[9:13]
        self.wB = navData[13:16]
        self.aB = navData[16:19]


    def update(self,navData, t):

        self.readNavData(navData)
        self.attitudeLoop(t)
        self.rateLoop()
        self.mixer()

        return self.actCmd


    def attitudeLoop(self,t):
        # Control inputs:
        #     accel_cmd
        #     yaw_cmd
        #     q_toBfromL

        accel_cmd = self.accel_cmd
        yaw_cmd = self.yaw_cmd
        q_toLfromB = self.q_toLfromB

        # Get body z vector expressed in L from
        R_toBfromL = quat2dcm( qConj(q_toLfromB ))
        bz = R_toBfromL[:,2]

        # Compute desired accel magnitude and direction in L frame
        aMag_cmd = np.sqrt( accel_cmd.dot(accel_cmd) )
        bz_cmd = accel_cmd / aMag_cmd # negate accel_cmd to get proper attitude

        # Reduced attitude command (thrust axis only)
        bzbz_cmd = bz.dot(bz_cmd)
        tilt = acos( np.clip(bzbz_cmd, -1.0, 1.0) ) # clip for cos domain error casuse floating point error
        rotAxis = skew3(bz).dot(bz_cmd)
        rotAxisMagSq = rotAxis.dot(rotAxis)
        if rotAxisMagSq == 0.0:
            rotAxis  = np.array([1.,0,0])
        else:
            rotAxis /= np.sqrt( rotAxisMagSq )

        # Limit max tilt angle
        tiltMax = 2*DEG2RAD
        if np.abs(tilt) > tiltMax:
            tilt = np.sign(tilt)*tiltMax
            qRed_err = axisAng2quat(rotAxis, tilt) # multiplicative error term (toCfromB)
            bz_cmd = qRot(qRed_err, bz)
        else:
            qRed_err = axisAng2quat(rotAxis, tilt) # multiplicative error term (toCfromB)

        qRed_toCfromL = skew4L(qRed_err).dot(qConj(q_toLfromB))

        # Full attitude command (considers yaw
        bx_cmd = np.array([ cos(yaw_cmd), sin(yaw_cmd), 0 ])
        by_cmd = skew3(bz_cmd).dot(bx_cmd)
        bx_cmd = skew3(by_cmd).dot(bz_cmd)
        qFull_toCfromL = dcm2quat( np.vstack(( bx_cmd, by_cmd, bz_cmd)).T )
        qFull_err = skew4L(qFull_toCfromL).dot(q_toLfromB) # multiplicative error term (toCfromB)

        # Mix Reduced and Full attitude commands
        pFrac = 0.2 # Full priority
        qMix = skew4L(qConj(qRed_toCfromL)).dot(qFull_toCfromL)
        qMix = qMix * np.sign(qMix[0]) # Force to shortest rotation due to quaternion double mapping
        angw = acos( np.clip(qMix[0], -1.0, 1.0) ) * pFrac
        angz = asin( np.clip(qMix[3], -1.0, 1.0) ) * pFrac
        qMix = np.array([cos(angw),0,0,sin(angz)])
        qMix_toCfromL = skew4L(qRed_toCfromL).dot(qMix)
        qMix_err = skew4L(qMix_toCfromL).dot(q_toLfromB)

        # Construct rate command
        qErr = qConj(qRed_err)
        qErr = qConj(qFull_err)
        qErr = qConj(qMix_err)

        qw = qErr[0]
        qv = qErr[1:]
        tau = 0.25
        if qw < 0.0:
            qv *= -1
        rateCmd = 2/tau*qv

        # Store output
        self.tilt           = tilt
        self.rotAxis        = rotAxis
        self.qRed_toCfromL  = qRed_toCfromL
        self.qFull_toCfromL = qFull_toCfromL
        self.qMix_toCfromL  = qMix_toCfromL
        self.qMix           = qMix
        self.qFull_err      = qFull_err
        self.qRed_err       = qRed_err
        self.qMix_err       = qMix_err
        self.qErr           = qErr
        self.rateCmd        = rateCmd


        if t ==10.:
            #print()
            #print(f'tilt = {tilt}')
            #print(f'rotAxis = {rotAxis}')
            #print(f'qRed_err = {qRed_err}')
            #print(f'qRed_toCfromL = {qRed_toCfromL}')
            #print(f'qFull_toCfromL = {qFull_toCfromL}')
            #print(f'qMix_toCfromL = {qMix_toCfromL}')
            #print()
            #breakpoint()
            pass


    def rateLoop(self):
        wB = self.wB
        self.rateErr = self.rateCmd - wB
        Kp = 50.0
        Kd = 0.1*Kp
        self.bodyCmd = Kp*self.rateErr + Kd*self.rateErr #wB

    def mixer(self):
        maxThrottle = 1000.0
        maxCmd = 1.00 * maxThrottle
        minCmd = 0.10 * maxThrottle

        #self.actCmd = np.array([1,0,0])
        cmdBx, cmdBy, cmdBz = self.bodyCmd
        throttle = 522.98471407


        # body2Act = np.vstack((np.zeros((3,3)),np.eye(3)))
        # self.actCmd = body2Act.dot(self.bodyCmd)

        body2Act = np.array([[ 0.,  1, -1],
                             [-1,  0,  1],
                             [ 0, -1, -1],
                             [ 1,  0,  1]])

        self.actCmd = throttle + body2Act.dot(self.bodyCmd)


        for k in range(4):
            self.cmd[k] = min( max(self.cmd[k], minCmd), maxCmd )




