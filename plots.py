from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

DEG2RAD = np.pi/180
RAD2DEG = 1/DEG2RAD

def plotQ_nav_vs_true(data):
	time = data['time']
	qTru = data['QtoBfromL']
	qNav = data['qNav']
	
	fig,axes = plt.subplots(4,1,sharex=True)
	for k,ax in enumerate(axes):
		ax.plot(time[0,:], qTru[k,:], 'r', label='True')
		ax.plot(time[0,:], qNav[k,:], 'b', label='Nav')
		ax.set_ylim([-1.1,1.1])
		ax.grid()
		ax.legend()
	axes[0].set_ylabel('qw')
	axes[1].set_ylabel('qx')
	axes[2].set_ylabel('qy')
	axes[3].set_ylabel('qz')
	axes[3].set_xlabel('Time(s)')



def plotData(data):
	rbody = data['rbody']
	time = data['time']
	
	fig,ax = plt.subplots(2,2,sharex=True)
	#ax[0,0].plot(data['time'][0,:], data['RofBwrtOexpB'][0,:],'r',label='rbx')
	#ax[0,0].plot(data['time'][0,:], data['RofBwrtOexpB'][1,:],'g',label='rby')
	#ax[0,0].plot(data['time'][0,:], data['RofBwrtOexpB'][2,:],'b',label='rbz')
	#ax[0,0].legend()
	#ax[0,0].grid()
	#ax[0,0].set_ylabel('RofBwrtOexpB')
	ax[1,0].plot(time, rbody['vel_B'][0,:],'r',label='vbx')
	ax[1,0].plot(time, rbody['vel_B'][1,:],'g',label='vby')
	ax[1,0].plot(time, rbody['vel_B'][2,:],'b',label='vbz')
	ax[1,0].legend()
	ax[1,0].grid()
	ax[1,0].set_ylabel('vel_B')
	ax[1,0].set_xlabel('Time(s)')
	ax[0,1].plot(time, rbody['pos_L'][0,:],'r',label='rLx')
	ax[0,1].plot(time, rbody['pos_L'][1,:],'g',label='rLy')
	ax[0,1].plot(time, rbody['pos_L'][2,:],'b',label='rLz')
	ax[0,1].legend()
	ax[0,1].grid()
	ax[0,1].set_ylabel('pos_L')
	ax[1,1].plot(time, rbody['vel_L'][0,:],'r',label='vLx')
	ax[1,1].plot(time, rbody['vel_L'][1,:],'g',label='vLy')
	ax[1,1].plot(time, rbody['vel_L'][2,:],'b',label='vLz')
	ax[1,1].legend()
	ax[1,1].grid()
	ax[1,1].set_ylabel('vel_L')
	ax[1,1].set_xlabel('Time(s)')	

	fig,ax = plt.subplots(2,1,sharex=True)
	ax[0].plot(time, rbody['qToBfromL'][0,:],'k',label='qw')
	ax[0].plot(time, rbody['qToBfromL'][1,:],'r',label='qx')
	ax[0].plot(time, rbody['qToBfromL'][2,:],'g',label='qy')
	ax[0].plot(time, rbody['qToBfromL'][3,:],'b',label='qz')
	ax[0].legend()
	ax[0].grid()
	ax[0].set_ylabel('Quaternion')
	ax[1].plot(time, rbody['wb'][0,:]*RAD2DEG,'r',label='wbx')
	ax[1].plot(time, rbody['wb'][1,:]*RAD2DEG,'g',label='wby')
	ax[1].plot(time, rbody['wb'][2,:]*RAD2DEG,'b',label='wbz')
	ax[1].legend()
	ax[1].grid()
	ax[1].set_ylabel('wb (deg/s)')
	ax[1].set_xlabel('Time(s)')

	fig,ax = plt.subplots(2,1,sharex=True)
	ax[0].plot(time, rbody['euler321_toBfromL'][0,:]*RAD2DEG,'r',label='yaw')
	ax[0].plot(time, rbody['euler321_toBfromL'][1,:]*RAD2DEG,'g',label='pitch')
	ax[0].plot(time, rbody['euler321_toBfromL'][2,:]*RAD2DEG,'b',label='roll')
	ax[0].legend()
	ax[0].grid()
	ax[0].set_ylabel('euler321_toBfromL (deg)')
	ax[1].plot(time, rbody['wb'][0,:]*RAD2DEG,'r',label='wbx')
	ax[1].plot(time, rbody['wb'][1,:]*RAD2DEG,'g',label='wby')
	ax[1].plot(time, rbody['wb'][2,:]*RAD2DEG,'b',label='wbz')
	ax[1].legend()
	ax[1].grid()
	ax[1].set_ylabel('wb (deg/s)')
	ax[1].set_xlabel('Time(s)')

def plotMotorData(data):
	
	time = data['time']
	motorList = [ name for name in data.keys() if 'motor' in name ]
	
	fig,ax = plt.subplots(2,2,sharex=True)
	ax[0,0].plot(time,data['motor1']['force'])
	ax[0,0].grid()
	ax[0,0].set_ylabel('Force (N)')
	ax[0,0].set_title('Motor 1')
	
	ax[0,1].plot(time,data['motor2']['force'])
	ax[0,1].grid()
	ax[0,1].set_ylabel('Force (N)')
	ax[0,1].set_title('Motor 2')	
	
	ax[1,0].plot(time,data['motor3']['force'])
	ax[1,0].grid()
	ax[1,0].set_ylabel('Force (N)')
	ax[1,0].set_title('Motor 3')	
	
	ax[1,1].plot(time,data['motor4']['force'])
	ax[1,1].grid()
	ax[1,1].set_ylabel('Force (N)')
	ax[1,1].set_title('Motor 4')	
	


def plotData3d(data):
	pos_L = data['rbody']['pos_L']

	rx = pos_L[0,:]
	ry = pos_L[1,:]
	rz = pos_L[2,:]
	
	maxVal = np.max(np.abs([rx,ry,rz]))
	
	

	fig = plt.figure()
	#ax = fig.add_axes([0, 0, 1, 1], projection='3d')
	ax = fig.gca(projection='3d')
	#ax.set_aspect("equal")
	
	ax.plot(rx,ry,rz,c='grey')
	ax.scatter(rx[0],ry[0],rz[0],c='g')
	ax.scatter(rx[-1],ry[-1],rz[-1],c='r')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.set_xlim([-maxVal, maxVal])
	ax.set_ylim([-maxVal, maxVal])
	ax.set_zlim([-maxVal, maxVal])
	
	
	
	
	
	