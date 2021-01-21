#!/usr/bin/env python3

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from Model6DOF import Model6DOF, ModelOutput, ModelQuadcopter
from EKF import AttitudeEKF
from ControlLoop import ControlLoop
from transform import *
from Plots import generate_plots

from IPython import embed as breakpoint

RAD2DEG = 180.0/np.pi
DEG2RAD = 1/RAD2DEG


showProgress  = True
showPlots     = True
wantAnimation = False

# Setup time
dt = 0.001 # sec
tStart = 0.0
tEnd = 10

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

control = ControlLoop(dt)

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
    'rMeas',
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
    'qMix',
    'qErr',
    'qFull_err',
    'qRed_err',
    'qMix_err',
]
controlData = ModelOutput(controlVarList, control)

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
for tk in time:


    progress = int(tk/tEnd*100)
    if progress % 10 == 0 and lastProgress != progress and showProgress:
        lastProgress = progress
        print(f'\t{progress}%')

    # Measurements
    wMeas = model.wMeas
    aMeas = model.aMeas
    mMeas = model.mMeas

    # EKF
    zMeas = np.hstack( (aMeas, mMeas) )
    ekf.predict(wMeas)
    ekf.update(zMeas)
    # Pass through measurments not yet implemented
    ekf.wb = model.wb

    # Control Loop
    if tk <= 1:
        control.accel_cmd = np.array([1.,1.,1.])
        control.yaw_cmd = 0.0 * DEG2RAD
    elif tk >= 5:
        control.accel_cmd = np.array([0,0,1.])
        control.yaw_cmd = 45.0 * DEG2RAD

    cmd = control.update(ekf,tk)

    # Update sim models
    model.update(tk,dt,cmd)

    # Store data
    modelData.append()
    controlData.append()
    ekfData.append()


# Post processing
time = np.asarray(time).T
modelData.process()
controlData.process()
ekfData.process()


# Plots
if showPlots:
    generate_plots(time,modelData,ekfData,controlData)
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

