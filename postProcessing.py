import numpy as np

from transform import quatNormalize, quat2euler321, quatConj, quatRot

DEG2RAD = np.pi / 180.0
RAD2DEG = 1/DEG2RAD

def post_calc(Y):
	# Express key parameters in L frame
	qToLfromB = quatConj(Y['rbody']['qToBfromL'])
	Y['rbody']['vel_L'] = quatRot(qToLfromB, Y['rbody']['vel_B'])
	
	# Euler angles
	Y['rbody']['euler321_toBfromL'] = quat2euler321(Y['rbody']['qToBfromL'])
	
	#import IPython; IPython.embed()

	return Y