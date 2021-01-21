#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
from math import radians, sin, cos, acos, asin

from transform import *

RAD2DEG = 180.0/np.pi
DEG2RAD = 1.0/RAD2DEG

def drawVec(ax, v, color):
    origin = np.zeros(3)
    vec = np.vstack((origin,v)).T
    ax.plot(vec[0,:], vec[1,:], vec[2,:],
            color=color)

# Summary
# qRot is working
# quat2dcm is working
# euler3212quat describes 321 rotation from stationary to moved frame (toBfromL)
# axisAng2quat is working
# quat2euler321 is backwards. Does not match euler3212quat

# Define Body Attitude in relation to L frame (toLfromB)
yaw   = 15
pitch = -20
roll  = 10
euler321_toBfromL = np.array([yaw,pitch,roll]) * DEG2RAD
q_toBfromL = euler3212quat( euler321_toBfromL )
R_toBfromL = quat2dcm(q_toBfromL)
Bx_expL_dcm = R_toBfromL[:,0]
By_expL_dcm = R_toBfromL[:,1]
Bz_expL_dcm = R_toBfromL[:,2]

# Define Command attitude (toCfromL)
yaw   = -15
pitch = 0
roll  = 0
euler321_toCfromL = np.array([yaw,pitch,roll]) * DEG2RAD
q_toCfromL = euler3212quat( euler321_toCfromL )
R_toCfromL = quat2dcm(q_toCfromL)
Cx_expL = R_toCfromL[:,0]
Cy_expL = R_toCfromL[:,1]
Cz_expL = R_toCfromL[:,2]

# Attitude from axis/angle
ang = -15
rotAxis = np.array([0,0,1.0])
q_toCfromL_axisAng = axisAng2quat(rotAxis, ang*DEG2RAD)
yaw_axisAng,pitch_axisAng,roll_axisAng = quat2euler321(q_toCfromL_axisAng)*RAD2DEG

# Attitude Command wrt Body
qErr = skew4L(qConj(q_toCfromL)).dot(q_toBfromL)
yaw,pitch,roll = quat2euler321(qErr)*RAD2DEG


# Setup vectors
Lx_expL = np.array([1, 0, 0.])
Ly_expL = np.array([0, 1, 0.])
Lz_expL = np.array([0, 0, 1.])
Bx = np.array([1, 0, 0.]) # Before rotation
By = np.array([0, 1, 0.]) # Before rotation
Bz = np.array([0, 0, 1.]) # Before rotation

# Rotate Body vectors to L from by q_toBfromL
Bx_expL = qRot( q_toBfromL, Bx)
By_expL = qRot( q_toBfromL, By)
Bz_expL = qRot( q_toBfromL, Bz)

Bx_expL_2 = R_toBfromL.dot(Bx)
By_expL_2 = R_toBfromL.dot(By)
Bz_expL_2 = R_toBfromL.dot(Bz)

# Print
#print(f'q_toBfromL         = {q_toBfromL}')
#print(f'q_toCfromL         = {q_toCfromL}')
#print(f'q_toCfromL_axisAng = {q_toCfromL_axisAng}')
#print(f'\tyaw   = {yaw_axisAng:0.2f}')
#print(f'\tpitch = {pitch_axisAng:0.2f}')
#print(f'\troll  = {roll_axisAng:0.2f}')
#print(f'qErr = {qErr}')
#print(f'\tyaw   = {yaw:0.2f}')
#print(f'\tpitch = {pitch:0.2f}')
#print(f'\troll  = {roll:0.2f}')


# Plot
#fig = plt.figure()
#ax = plt3.Axes3D(fig)
#ax.plot(0,0,0,'ko')
#drawVec(ax, Lx_expL, 'black')
#drawVec(ax, Ly_expL, 'black')
#drawVec(ax, Lz_expL, 'black')
#drawVec(ax, Bx_expL, 'red')        # qRot(q_toBfromL)
#drawVec(ax, By_expL, 'red')        # qRot(q_toBfromL)
#drawVec(ax, Bz_expL, 'red')        # qRot(q_toBfromL)
#drawVec(ax, Bx_expL_2, 'blue')     # R_toBfromL.dot()
#drawVec(ax, By_expL_2, 'blue')     # R_toBfromL.dot()
#drawVec(ax, Bz_expL_2, 'blue')     # R_toBfromL.dot()
#drawVec(ax, Bx_expL_dcm, 'orange') # R_toBfromL vectors
#drawVec(ax, By_expL_dcm, 'orange') # R_toBfromL vectors
#drawVec(ax, Bz_expL_dcm, 'orange') # R_toBfromL vectors
#drawVec(ax, Cx_expL, 'green')      # R_toCfromL vectors
#drawVec(ax, Cy_expL, 'green')      # R_toCfromL vectors
#drawVec(ax, Cz_expL, 'green')      # R_toCfromL vectors
#
#ax.view_init(elev=-90, azim=-90)
#ax.set_xlim(-1,1)
#ax.set_ylim(-1,1)
#ax.set_zlim(-1,1)
#plt.show()




# AttitudeLoop
accel_cmd = np.array([0,0,1.]) # negate accel_cmd to get proper attitude
yaw_cmd = 10 * DEG2RAD

yaw   = 0
pitch = 10
roll  = 0
euler321_toBfromL = np.array([yaw,pitch,roll]) * DEG2RAD
q_toLfromB = qConj(euler3212quat( euler321_toBfromL ))

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
qRed_err = axisAng2quat(rotAxis, tilt) # multiplicative error term (toCfromB)
qRed_toCfromL = skew4L(qRed_err).dot(qConj(q_toLfromB))

# Full attitude command (considers yaw
bx_cmd = np.array([ cos(yaw_cmd), sin(yaw_cmd), 0 ])
by_cmd = skew3(bz_cmd).dot(bx_cmd)
bx_cmd = skew3(by_cmd).dot(bz_cmd)
qFull_toCfromL = dcm2quat( np.vstack(( bx_cmd, by_cmd, bz_cmd)).T )
qFull_err = skew4L(qFull_toCfromL).dot(q_toLfromB) # multiplicative error term (toCfromB)

# Mix Reduced and Full attitude commands
pFrac = 0.5 # Full priority
qMix = skew4L(qConj(qRed_toCfromL)).dot(qFull_toCfromL)
qMix = qMix * np.sign(qMix[0])
angw = acos( np.clip(qMix[0], -1.0, 1.0) ) * pFrac
angz = asin( np.clip(qMix[3], -1.0, 1.0) ) * pFrac
qMix = np.array([cos(angw),0,0,sin(angz)])
qMix_toCfromL = skew4L(qRed_toCfromL).dot(qMix)
qMix_err = skew4L(qMix_toCfromL).dot(q_toLfromB)

# Construct rate command
qErr = qConj(qRed_err)


# Setup vectors
Lx_expL = np.array([1, 0, 0.])
Ly_expL = np.array([0, 1, 0.])
Lz_expL = np.array([0, 0, 1.])
Bx = np.array([1, 0, 0.]) # Before rotation
By = np.array([0, 1, 0.]) # Before rotation
Bz = np.array([0, 0, 1.]) # Before rotation

# Rotate Body vectors to L from by q_toBfromL
Bx_expL = qRot( qConj(q_toLfromB), Bx)
By_expL = qRot( qConj(q_toLfromB), By)
Bz_expL = qRot( qConj(q_toLfromB), Bz)

# Cred
Rred_toCfromL = quat2dcm(qRed_toCfromL)
CredX_expL = Rred_toCfromL[:,0]
CredY_expL = Rred_toCfromL[:,1]
CredZ_expL = Rred_toCfromL[:,2]

# Cmix
Rfull_toCfromL = quat2dcm(qFull_toCfromL)
CfullX_expL = Rfull_toCfromL[:,0]
CfullY_expL = Rfull_toCfromL[:,1]
CfullZ_expL = Rfull_toCfromL[:,2]

# Cmix
Rmix_toCfromL = quat2dcm(qMix_toCfromL)
CmixX_expL = Rmix_toCfromL[:,0]
CmixY_expL = Rmix_toCfromL[:,1]
CmixZ_expL = Rmix_toCfromL[:,2]

# Print
print(f'qRed_toCfromL  = {qRed_toCfromL}')
print(f'qFull_toCfromL = {qFull_toCfromL}')
print(f'qMix_toCfromL  = {qMix_toCfromL}')
print(f'qRed_err       = {qRed_err}')
print(f'qFull_err      = {qFull_err}')
print(f'qMix_err       = {qMix_err}')
print(f'qMix           = {qMix}')
yaw,pitch,roll = quat2euler321(qMix)*RAD2DEG
print(f'\tyaw   = {yaw:0.2f}')
print(f'\tpitch = {pitch:0.2f}')
print(f'\troll  = {roll:0.2f}')

# Scale for plots
CredX_expL  *= 0.85
CredY_expL  *= 0.85
CredZ_expL  *= 0.85
CfullX_expL *= 0.9
CfullY_expL *= 0.9
CfullZ_expL *= 0.9
CmixX_expL *= 0.95
CmixY_expL *= 0.95
CmixZ_expL *= 0.95

# Plot
fig = plt.figure()
ax = plt3.Axes3D(fig)
drawVec(ax, bz_cmd, 'green')
drawVec(ax, Lx_expL, 'black')
drawVec(ax, Ly_expL, 'black')
drawVec(ax, Lz_expL, 'black')
drawVec(ax, Bx_expL, 'red')        # qRot(q_toBfromL)
drawVec(ax, By_expL, 'red')        # qRot(q_toBfromL)
drawVec(ax, Bz_expL, 'red')        # qRot(q_toBfromL)
drawVec(ax, CredX_expL, 'blue')
drawVec(ax, CredY_expL, 'blue')
drawVec(ax, CredZ_expL, 'blue')
drawVec(ax, CfullX_expL, 'cyan')
drawVec(ax, CfullY_expL, 'cyan')
drawVec(ax, CfullZ_expL, 'cyan')
drawVec(ax, CmixX_expL, 'orange')
drawVec(ax, CmixY_expL, 'orange')
drawVec(ax, CmixZ_expL, 'orange')

ax.plot(bz_cmd[0],bz_cmd[1],bz_cmd[2],'go')
ax.plot(CredZ_expL[0],CredZ_expL[1],CredZ_expL[2],'o', color = 'blue')
ax.plot(CfullZ_expL[0],CfullZ_expL[1],CfullZ_expL[2],'o', color = 'cyan')
ax.plot(CmixZ_expL[0],CmixZ_expL[1],CmixZ_expL[2],'o', color = 'orange')
ax.plot(1,0,0,'k.')
ax.plot(0,0,0,'k*',markersize=10)
ax.view_init(azim=35, elev=-150)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.show()
