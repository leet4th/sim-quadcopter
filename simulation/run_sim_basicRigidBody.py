#!/usr/bin/env python3

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

from Model6DOF import Model6DOF, ModelOutput, ModelQuadcopter
from EKF import AttitudeEKF
from ControlLoop import ControlLoop
from transform import *

from IPython import embed as breakpoint

RAD2DEG = 180.0/np.pi
DEG2RAD = 1/RAD2DEG

# Setup time
dt = 0.001 # sec
tStart = 0.0
tEnd = 25

nSteps = int((tEnd-tStart)/dt) + 1
time,dt = np.linspace(tStart, tEnd, num=nSteps, retstep=True)

# Setup sim model
model = Model6DOF(dt)


control = ControlLoop(dt)


qStart = euler3212quat(np.array([0,0,0])*DEG2RAD)
model.state[6:10] = qStart

# Setup model outputs
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
    'euler321_toBfromL',
    'euler321',
]
modelData = ModelOutput(modelVarList, model)
controlVarList = [
    'actCmd',
    'bodyCmd',
    'rateErr',
    'rateCmd',
    'tilt',
    'rotAxis',
    'qRed_toCfromL',
    'qFull_toCfromL',
    'qMix_toCfromL',
    'qErr',
    'qFull_toCfromB',
    'qRed_toCfromB',
]
controlData = ModelOutput(controlVarList, control)

lastProgress = -1
showProgress = True
showPlots = True
wantAnimation = True
for tk in time:

    progress = int(tk/tEnd*100)
    if progress % 10 == 0 and lastProgress != progress and showProgress:
        lastProgress = progress
        print(f'\t{progress}%')


    control.accel_cmd = np.array([0,1.,1.])
    control.yaw_cmd = 0.0 * DEG2RAD

    cmd = control.update(model,tk)

    # Update sim models
    model.update(tk,dt,cmd)

    # Store data
    modelData.append()
    controlData.append()


# Post processing
time = np.asarray(time).T
modelData.process()
controlData.process()


# Plots
if showPlots:

    fig,ax = plt.subplots(1,1,sharex=True)
    ax.plot(time,modelData.data['q_toLfromB'][0,:],'k',label='qw')
    ax.plot(time,modelData.data['q_toLfromB'][1,:],'r',label='qx')
    ax.plot(time,modelData.data['q_toLfromB'][2,:],'g',label='qy')
    ax.plot(time,modelData.data['q_toLfromB'][3,:],'b',label='qz')
    ax.grid()
    ax.legend()
    ax.set_ylim([-1.1,1.1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q_toLfromB')

    fig,ax = plt.subplots(1,1,sharex=True)
    ax.plot(time,modelData.data['wb'][0,:]*RAD2DEG,'r',label='wbx')
    ax.plot(time,modelData.data['wb'][1,:]*RAD2DEG,'g',label='wby')
    ax.plot(time,modelData.data['wb'][2,:]*RAD2DEG,'b',label='wbz')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('wb (deg/s)')

    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(time,controlData.data['rateErr'][0,:]*RAD2DEG,'r',label='rateErr-x')
    ax[0].plot(time,controlData.data['rateErr'][1,:]*RAD2DEG,'g',label='rateErr-y')
    ax[0].plot(time,controlData.data['rateErr'][2,:]*RAD2DEG,'b',label='rateErr-z')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_ylabel('rateErr (deg/s)')
    ax[1].plot(time,controlData.data['bodyCmd'][0,:]*RAD2DEG,'r',label='bodyCmd-x')
    ax[1].plot(time,controlData.data['bodyCmd'][1,:]*RAD2DEG,'g',label='bodyCmd-y')
    ax[1].plot(time,controlData.data['bodyCmd'][2,:]*RAD2DEG,'b',label='bodyCmd-z')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_ylabel('bodyCmd (deg/s)')
    ax[1].set_xlabel('Time (s)')

    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(time,controlData.data['tilt'][:]*RAD2DEG,'r')
    ax[0].grid()
    ax[0].set_ylim(ymin=-0.5)
    ax[0].set_ylabel('Tilt (deg)')
    ax[1].plot(time,controlData.data['qRed_toCfromB'][0,:], 'r', label='qRed_toCfromL')
    ax[1].plot(time,controlData.data['qFull_toCfromB'][0,:],'g', label='qFull_toCfromL')
    for k in range(1,4):
        ax[1].plot(time,controlData.data['qRed_toCfromB'][k,:], 'r')
        ax[1].plot(time,controlData.data['qFull_toCfromB'][k,:],'g')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_ylim(ymin=-0.5)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Attitude Error (toCfromB)')

    fig,ax = plt.subplots(1,1,sharex=True)
    ax.plot(time,controlData.data['qRed_toCfromL'][0,:], 'r', label='qRed_toCfromL')
    ax.plot(time,controlData.data['qFull_toCfromL'][0,:],'g', label='qFull_toCfromL')
    #ax.plot(time,controlData.data['qMix_toCfromL'][0,:], 'b', label='qMix_toCfromL')
    for k in range(1,4):
        ax.plot(time,controlData.data['qRed_toCfromL'][k,:], 'r')
        ax.plot(time,controlData.data['qFull_toCfromL'][k,:],'g')
        #ax.plot(time,controlData.data['qMix_toCfromL'][k,:], 'b')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Attitude Command (toCfromL)')


    #fig,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(time,modelData.data['euler321'][0,:]*RAD2DEG,'r',label='yaw')
    #ax.plot(time,modelData.data['euler321'][1,:]*RAD2DEG,'g',label='pitch')
    #ax.plot(time,modelData.data['euler321'][2,:]*RAD2DEG,'b',label='roll')
    #ax.plot(time,modelData.data['q_toLfromB'][1,:]*2*RAD2DEG,'b--',label='2*qx')
    #ax.plot(time,modelData.data['q_toLfromB'][2,:]*2*RAD2DEG,'g--',label='2*qy')
    #ax.plot(time,modelData.data['q_toLfromB'][3,:]*2*RAD2DEG,'r--',label='2*qz')
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('euler321 (toLfromB)')

    #fig,ax = plt.subplots(1,1,sharex=True)
    #ax.plot(time,modelData.data['euler321_toBfromL'][0,:]*RAD2DEG,'r',label='yaw')
    #ax.plot(time,modelData.data['euler321_toBfromL'][1,:]*RAD2DEG,'g',label='pitch')
    #ax.plot(time,modelData.data['euler321_toBfromL'][2,:]*RAD2DEG,'b',label='roll')
    #ax.grid()
    #ax.legend()
    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('euler321 (toBfromL)')


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
        for q_toBfromL in modelData.data['q_toBfromL'][:,::100].T:
            pv.clock.tick(int(model.dt*50*1000))

            # sys update attitude
            block.sys.setQuat(q_toBfromL)

            # display updated attitude
            pv.display()
            pygame.display.flip()




