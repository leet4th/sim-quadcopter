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

# Time step
dt         = 0.001 # sec
dt_ekf     = 0.01  # sec
dt_control = 0.01  # sec

# Setup time
tStart = 0.0
tEnd = 15

tmod_ekf = int(dt_ekf/dt)
tmod_control = int(dt_control/dt)

# Setup sim model
model = ModelQuadcopter(dt)
# Sensor Models
model.gyroBias   *= 1
model.gyroNoise  *= 1
model.accelBias  *= 1
model.accelNoise *= 1
model.magBias    *= 1
model.magNoise   *= 1
model.gpsBias    *= 1
model.gpsNoise   *= 1

# Setup EKF
ekf = AttitudeEKF(dt_ekf)

# Control
control = ControlLoop(dt_control)

# Starting Conditions
qStart = euler3212quat(np.array([0,0,0])*DEG2RAD)
wStart = np.array([0, 0, 0]) * DEG2RAD

model.state[6:10] = qStart
model.state[10:13] = wStart
#ekf.state[6:10] = qStart

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
    'aL',
    'wMeas',
    'aMeas',
    'wB',
    'aB',
]
ekfData = ModelOutput(ekfVarList,ekf)

k = 0
tk = tStart
time = []
nProg = 10
progStep = int(1/nProg*100)
progress = 0
tmod_progress = int(tEnd/10/dt)
while tk <= tEnd:

    if (k % tmod_progress) == 0 and showProgress:
        print(f'\t{progress}%')
        progress += progStep


    # Nav
    if (k % tmod_ekf) == 0:
        # Measurements
        wMeas = model.wMeas
        aMeas = model.aMeas
        mMeas = model.mMeas
        rMeas = model.rMeas

        # EKF
        zMeas = np.hstack( (rMeas, aMeas, mMeas) )
        ekf_u = np.hstack( (wMeas, aMeas) )
        navData = ekf.updateNav(ekf_u, zMeas)

    # Control Loop
    if (k % tmod_control) == 0:
        control.accel_cmd = np.array([1.,0.,1.])
        control.yaw_cmd = 0.0 * DEG2RAD
        cmd = control.update(navData,tk)

    # Update sim models
    model.update(tk,dt,cmd)

    # Store data
    time.append(tk)
    modelData.append()
    controlData.append()
    ekfData.append()

    # Update for next iteration
    k += 1
    tk += dt


# Post processing
time = np.asarray(time)
modelData.process()
controlData.process()
ekfData.process()

ekfData.data['r_BwrtLexpL'] = ekfData.data['state'][0:3,:]
ekfData.data['v_BwrtLexpL'] = ekfData.data['state'][3:6,:]
ekfData.data['q_toLfromB'] = ekfData.data['state'][6:10,:]
ekfData.data['a_BwrtLexpL'] = ekfData.data['aL'][:]

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
        #for q_toBfromL in ekfData.data['state'][6:10,::100].T:
        for q_toBfromL in ekfData.data['state'][6:10,::100].T:
            pv.clock.tick(int(model.dt*50*1000))

            # sys update attitude
            block.sys.setQuat(q_toBfromL)

            # display updated attitude
            pv.display()
            pygame.display.flip()

