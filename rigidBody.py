import numpy as np
from transform import *
from solver import *

#from transfrom import dQuat, skew


class test_class:
	def __init__(self, input1=np.array([1,2,3]) ):
		self.input1 = input1
	
test = test_class()

def rbody_calc(X,U,**kwargs):
	"""
	Rigid body model
	Integrate states in body frame
	
	INPUTS:
		t = current time
		X = State vector
			0  = RofBwrtOexpB_x
			1  = RofBwrtOexpB_y
			2  = RofBwrtOexpB_z
			3  = VofBwrtOexpB_x
			4  = VofBwrtOexpB_y
			5  = VofBwrtOexpB_z
			6  = QtoBfromL_w
			7  = QtoBfromL_x
			8  = QtoBfromL_y
			9  = QtoBfromL_z
			10 = wb_x
			11 = wb_y
			12 = wb_z
		U = Input vector
			0  = Fb_x
			1  = Fb_y
			2  = Fb_z
			3  = Mb_x
			4  = Mb_y
			5  = Mb_z
		*ars=const
			const{'mass',J'}
	OUTPUTS:
		XDOT = dState vector
			0  = VofBwrtOexpB_x
			1  = VofBwrtOexpB_y
			2  = VofBwrtOexpB_z
			3  = VDOTofBwrt0expB_x
			4  = VDOTofBwrt0expB_y
			5  = VDOTofBwrt0expB_z
			6  = QDOTtoBfromL_w
			7  = QDOTtoBfromL_x
			8  = QDOTtoBfromL_y
			9  = QDOTtoBfromL_z
			10 = wDOTb_x
			11 = wDOTb_y
			12 = wDOTb_z
	"""
	
	#import IPython; IPython.embed()
	

	# Unpack state vector
	RofBwrtOexpL = np.array( X[0:3] )
	VofBwrtOexpB = np.array( X[3:6] )
	QtoBfromL = np.array( X[6:10] )
	wb = np.array( X[10:13] )
	
	# Unpack Input vector
	Fb  = np.array( U[0:3] )
	Mb = np.array( U[3:6] )
	
	# Unpack constants
	mass = kwargs['mass']
	J = kwargs['J']
	gravL = np.array([0,0,kwargs['g']])
	
	# Linear equation of motion
	vDot_b = Fb/mass + quatRot(QtoBfromL, gravL) - np.matmul(skew(wb),VofBwrtOexpB)
	
	# Rotational equation of motion
	J_wDot = Mb - np.matmul( skew(wb), np.matmul(J,wb) )
	wDot = np.matmul(np.linalg.inv(J), J_wDot)
	
	# Solve for rate of change of quaternion 
	qDot = dQuat(QtoBfromL, wb)
	
	# Build dState vector
	XDOT = np.zeros(13)
	XDOT[0:3] = quatRot(quatConj(QtoBfromL), VofBwrtOexpB) #VofBwrtOexpL
	XDOT[3:6] = vDot_b
	XDOT[6:10] = qDot
	XDOT[10:13] = wDot
	
	return XDOT

	
	
	
	
	
	



