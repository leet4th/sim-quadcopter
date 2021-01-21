#!/usr/bin/env python3

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

from Model6DOF import Model6DOF, ModelOutput, ModelQuadcopter
from EKF import AttitudeEKF
from transform import *

from IPython import embed as breakpoint

RAD2DEG = 180.0/np.pi
DEG2RAD = 1/RAD2DEG



# Setup time
dt = 0.001 # sec
tStart = 0.0
tEnd = 10.0

nSteps = int((tEnd-tStart)/dt) + 1
time,dt = np.linspace(tStart, tEnd, num=nSteps, retstep=True)

# Setup sim model
model = ModelQuadcopter(dt)
# Noise
model.gyroNoise  *= 1
model.accelNoise *= 1
model.magNoise   *= 1

qStart = euler3212quat(np.array([0,0,0])*DEG2RAD)
model.state[6:10] = qStart

# Setup EKF
ekf = AttitudeEKF(dt)


## Fbx,Fby,Fbz,Mbx,Mby,Mbz
#cmd = np.array([
#    0,
#    0,
#    0,
#    1000,
#    1000,
#    1000,
#])

# wm1,wm2,wm3,wm4
cmd = np.array([
    0,
    0,
    0,
    0,
])
cmd = np.ones(4) * np.sqrt(model.mass * model.g / 4 )

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


    fix,ax = plt.subplots(1,1,sharex=True)
    ax.plot(time,modelData.data['r_BwrtLexpL'][0,:],label='x')
    ax.plot(time,modelData.data['r_BwrtLexpL'][1,:],label='y')
    ax.plot(time,modelData.data['r_BwrtLexpL'][2,:],label='z')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('r_BwrtLexpL')

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




