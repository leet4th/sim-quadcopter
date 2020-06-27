import numpy as np
import time as tm
import matplotlib.pyplot as plt

# Sim
from sim.simModels import rigidBodyDynamics, motor, motor2body, mpu9250
from sim.utilities import setupTime, saveData, updateModels, getModelOutput
from sim.postProcessing import post_calc
from sim.plots import plotData, plotData3d, plotMotorData

# Flt
from flt.control import quadControl

# Simulation time
dt = 0.001
tStart = 0
tEnd = 1
time,dt,N = setupTime(tStart,tEnd,dt)

"""
rigidBodyDynamics
"""
# rbody
# Constants
mass = 0.468 # kg
gravity = 9.81 # m/s^2
inertia = np.eye(3)
inertia[0,0] = 4.856E-3 # Ixx [kg*m^2]
inertia[1,1] = 4.856E-3 # Iyy [kg*m^2]
inertia[2,2] = 8.801E-3 # Izz [kg*m^2]
# Initial States
pos_L       = np.array([0,0,0])
vel_B       = np.array([1,0,0])
qToBfromL   = np.array([1,0,0,0])
wb          = np.array([0,0,0.5])*np.pi/180

"""
motor
"""
# motor parameters
motor_w   = 0        # Initial motor speed
motor_Kf  = 2.980E-6 # Force Coefficent
motor_Kt  = 1.140E-7 # Torque Coefficent
motor_tau = 0.0001    # Speed time constant
armLength  = 0.225   # Arm Length [m]
motor1_dir =  1      # Motor 1 Spin Direction +1 = CCW, -1 CW (top looking down)
motor2_dir = -1      # Motor 2 Spin Direction +1 = CCW, -1 CW (top looking down)
motor3_dir =  1      # Motor 3 Spin Direction +1 = CCW, -1 CW (top looking down)
motor4_dir = -1      # Motor 4 Spin Direction +1 = CCW, -1 CW (top looking down)
motor1_loc = np.array([ armLength,         0, 0]) # Motor 1 Location in body frame
motor2_loc = np.array([         0, armLength, 0]) # Motor 2 Location in body frame
motor3_loc = np.array([-armLength,         0, 0]) # Motor 3 Location in body frame
motor4_loc = np.array([         0,-armLength, 0]) # Motor 4 Location in body frame
motor1_vec = np.array([0,0,-1]) # Motor 1 Pointing vector in body frame
motor2_vec = np.array([0,0,-1]) # Motor 2 Pointing vector in body frame
motor3_vec = np.array([0,0,-1]) # Motor 3 Pointing vector in body frame
motor4_vec = np.array([0,0,-1]) # Motor 4 Pointing vector in body frame

"""
mpu9250 (imu)
"""
accel_3sig = 0.1
gyro_3sig = 0.1
mag_3sig = 0.1

"""
Initialize models
"""
# Initialize motors
motor1 = motor(motor_w, motor1_dir, motor_Kf, motor_Kt, motor_tau, motor1_loc, motor1_vec)
motor2 = motor(motor_w, motor2_dir, motor_Kf, motor_Kt, motor_tau, motor2_loc, motor2_vec)
motor3 = motor(motor_w, motor3_dir, motor_Kf, motor_Kt, motor_tau, motor3_loc, motor3_vec)
motor4 = motor(motor_w, motor4_dir, motor_Kf, motor_Kt, motor_tau, motor4_loc, motor4_vec)
# Motor mapping in body frame
motorMap = motor2body([motor1, motor2, motor3, motor4])
# Initialize rbody
rbody = rigidBodyDynamics( pos_L, vel_B, qToBfromL, wb, mass, inertia, gravity)
# Initialize imu
imu = mpu9250(rbody, accel_3sig, gyro_3sig, mag_3sig)

"""
Control
"""
# Commands
cmd1 = 0
cmd2 = 0
cmd3 = 0
cmd4 = 0

"""
Setup models dict
"""
models = {}
models['rbody'] = {}
models['rbody']['model'] = rbody
models['rbody']['input'] = [motorMap.getBodyForceMoment]
models['motor1'] = {}
models['motor1']['model'] = motor1
models['motor1']['input'] = [cmd1]
models['motor2'] = {}
models['motor2']['model'] = motor2
models['motor2']['input'] = [cmd2]
models['motor3'] = {}
models['motor3']['model'] = motor3
models['motor3']['input'] = [cmd3]
models['motor4'] = {}
models['motor4']['model'] = motor4
models['motor4']['input'] = [cmd4]

"""
Run Simulation
"""
tic = tm.perf_counter()
N_update = 10
k_update = 1
print('\n\nStarting simulation...')
for k,tk in enumerate(time):

	pos_L       = np.array( rbody.state[ 0:3 ] )
	vel_B       = np.array( rbody.state[ 3:6 ] )
	qToBfromL   = np.array( rbody.state[ 6:10] )
	wb          = np.array( rbody.state[10:13] )


	cmd = 0
	cmd += 625
	cmd += (0 - pos_L[2]) * -0.5
	cmd += (0 - vel_B[2]) * -5
	
	cmd = np.clip(cmd,400,800)

	# Update control
	cmd1 = cmd
	cmd2 = cmd
	cmd3 = cmd
	cmd4 = cmd
	
	# Set command
	models['motor1']['input'] = [cmd1]
	models['motor2']['input'] = [cmd2]
	models['motor3']['input'] = [cmd3]
	models['motor4']['input'] = [cmd4]

	# Integrate models for time step
	updateModels(models,dt)
	
	if k > k_update/N_update*N:
		print(f"\tSim {int(tk/tEnd*100)}% Complete")
		k_update += 1
# Post processing
data = getModelOutput(models, time)
data = post_calc(data)
print('\tSim 100% Complete')
toc = tm.perf_counter()
print(f"Completed in {toc-tic} seconds\n\n")



plotData(data)
plotMotorData(data)
plotData3d(data)
plt.show()



