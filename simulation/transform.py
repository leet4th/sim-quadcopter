import numpy as np

def dcm2euler321(dcm):
	yaw = np.arctan2(dcm[0,1], dcm[0,0])
	pitch = -np.arcsin( dcm[0,2] )
	roll = np.arctan2( dcm[1,2], dcm[2,2] )
	
	ang = np.array([yaw, pitch, roll])
	
	return ang

def dcm2quat(dcm):
	"""
	Determine quaternion corresponding to dcm using
	the stanley method. 
	
	Flips sign to always return shortest path quaterion
	so w >= 0
	
	Converts the 3x3 DCM into the quaterion where the 
	first component is the real part
	"""
	
	tr = np.trace(dcm)
	
	w = 0.25*(1+tr)
	x = 0.25*(1+2*dcm[0,0]-tr)
	y = 0.25*(1+2*dcm[1,1]-tr)
	z = 0.25*(1+2*dcm[2,2]-tr)
	
	kMax = np.argmax([w,x,y,z])
	
	if kMax == 0:
		w = np.sqrt(w)
		x = 0.25*(dcm[1,2]-dcm[2,1])/w
		y = 0.25*(dcm[2,0]-dcm[0,2])/w
		z = 0.25*(dcm[0,1]-dcm[1,0])/w
	
	elif kMax == 1:
		x = np.sqrt(x)
		w = 0.25*(dcm[1,2]-dcm[2,1])/x
		if w<0:
			x = -x
			w = -w
		y = 0.25*(dcm[0,1]+dcm[1,0])/x
		z = 0.25*(dcm[2,0]+dcm[0,2])/x
		
	elif kMax == 2:
		y = np.sqrt(y)
		w = 0.25*(dcm[2,0]-dcm[0,2])/y
		if w<0:
			y = -y
			w = -w
		x = 0.25*(dcm[0,1]+dcm[1,0])/y
		z = 0.25*(dcm[1,2]+dcm[2,1])/y
		
	elif kMax == 3:
		z = np.sqrt(z)
		w = 0.25*(dcm[0,1]-dcm[1,0])/z
		if w<0:
			z = -z
			w = -w
		x = 0.25*(dcm[2,0]+dcm[0,2])/z
		y = 0.25*(dcm[1,2]+dcm[2,1])/z
		
	# Ensure unit quaternion
	q = normalize_fast( np.array([w,x,y,z]) )
	
	return q
	
def euler3212dcm(ang):
	"""
	Converts euler321 to dcm
	"""
	
	c1 = np.cos(ang[0])
	s1 = np.sin(ang[0])
	c2 = np.cos(ang[1])
	s2 = np.sin(ang[1])
	c3 = np.cos(ang[2])
	s3 = np.sin(ang[2])
	
	# Build Direction Cosine Matrix (DCM)
	dcm = np.zeros((3,3))
	dcm[0,0] = c2*c1
	dcm[0,1] = c2*s1
	dcm[0,2] = -s2
	dcm[1,0] = s3*s2*c1 - c3*s1
	dcm[1,1] = s3*s2*s1 + c3*c1
	dcm[1,2] = s3*c2
	dcm[2,0] = c3*s2*c1 + s3*s1
	dcm[2,1] = c3*s2*s1 - s3*c1
	dcm[2,2] = c3*c2

	return dcm

def euler3212quat(ang):
	"""
	Converts the euler321 sequence to quaternion attitude representation
	
	euler321 refers to three sequential rotations about the axes specified by the number
	
	euler321 -> RotZ(ang1) -> RotY(ang2) -> RotX(ang3)
	
	This maps to the typical aerospace convention of
	euler321 -> RotZ(yaw) -> RotY(pitch) -> RotX(roll)
	
	"""
	
	c1 = np.cos(ang[0]/2)
	s1 = np.sin(ang[0]/2)
	c2 = np.cos(ang[1]/2)
	s2 = np.sin(ang[1]/2)
	c3 = np.cos(ang[2]/2)
	s3 = np.sin(ang[2]/2)
	
	w = c1*c2*c3+s1*s2*s3
	x = c1*c2*s3-s1*s2*c3
	y = c1*s2*c3+s1*c2*s3
	z = s1*c2*c3-c1*s2*s3
	
	# Ensure unit quaternion
	q = quatNormalize( np.array([w,x,y,z]) )
	
	return q	

def quat2dcm(q):
	"""
	Convert quaternion to DCM
	"""
	
	# Extract components
	w = q[0]
	x = q[1]
	y = q[2]
	z = q[3]
	
	# Reduce repeated calculations
	ww = w*w
	xx = x*x
	yy = y*y
	zz = z*z  
	wx = w*x
	wy = w*y
	wz = w*z
	xy = x*y
	xz = x*z
	yz = y*z	
	
	# Build Direction Cosine Matrix (DCM)
	dcm = np.zeros((3,3))
	dcm[0,0] = ww + xx - yy - zz
	dcm[0,1] = 2*(xy + wz)
	dcm[0,2] = 2*(xz - wy)
	dcm[1,0] = 2*(xy - wz)
	dcm[1,1] = ww - xx + yy - zz
	dcm[1,2] = 2*(yz + wx)
	dcm[2,0] = 2*(xz + wy)
	dcm[2,1] = 2*(yz - wx)
	dcm[2,2] = ww - xx - yy + zz
	
	return dcm	

def quat2euler321(q):

	# Extract components
	w = q[0]
	x = q[1]
	y = q[2]
	z = q[3]
	
	# Reduce repeated calculations
	ww = w*w
	xx = x*x
	yy = y*y
	zz = z*z  
	wx = w*x
	wy = w*y
	wz = w*z
	xy = x*y
	xz = x*z
	yz = y*z
	
	# Calculate angles	
	yaw   = np.arctan2( 2*(xy + wz), ww + xx - yy - zz );
	pitch = -np.arcsin(2*(xz - wy));
	roll  = np.arctan2(2*(yz + wx) , ww - xx - yy + zz);

	ang = np.array([yaw, pitch, roll])

	return ang

def calcBfromQ(q):
	"""
	dq/dt = 1/2 [B(q)] w
	"""
	w = q[0]
	x = q[1]
	y = q[2]
	z = q[3]

	B = np.zeros((4,3))
	B[0,0] = -x
	B[0,1] = -y
	B[0,2] = -z
	B[1,0] =  w
	B[1,1] = -z
	B[1,2] =  y
	B[2,0] =  z
	B[2,1] =  w
	B[2,2] = -x
	B[3,0] = -y
	B[3,1] =  x
	B[3,2] =  w

	return B

def dQuat(q,w):
	"""
	dq/dt = 1/2 [B(q)] w
	"""
	return 0.5 * np.matmul(calcBfromQ(q), w)
	
def skew(x):
	"""
	Returns skew symetric matic of x
	where x is a vector of length 3
	"""
	xSkew = np.zeros((3,3))
	xSkew[0,1] = -x[2]
	xSkew[0,2] =  x[1]
	xSkew[1,0] =  x[2]
	xSkew[1,2] = -x[0]
	xSkew[2,0] = -x[1]
	xSkew[2,1] =  x[0]
	return xSkew

def quatNormalize(q):
	normSquared = 0
	for each in q:
		normSquared += each*each
	q = q / np.sqrt(normSquared)
	return q
	
def quatNormalize_fast(q):
	EPS = 7./3 - 4./3 -1
	normSquared = 0
	for each in q:
		normSquared += each*each
	if abs(normSquared - 1) > (4*EPS):
		q = q / np.sqrt(normSquared)
	return q
	
def quatConj(q):
	q_out = np.array(q, copy=True)
	q_out[1:] = -q[1:]
	
	return q_out

def quatMult(a, b):
	"""
	Performs successive rotations via quaternion
	
	Qca = Qcb * Qba
	
	Output corresponds to successive rotations a and b
	"""

	w = b[0]*a[0]-b[1]*a[1]-b[2]*a[2]-b[3]*a[3]
	x = b[1]*a[0]+b[0]*a[1]+b[3]*a[2]-b[2]*a[3]
	y = b[2]*a[0]-b[3]*a[1]+b[0]*a[2]+b[1]*a[3]
	z = b[3]*a[0]+b[2]*a[1]-b[1]*a[2]+b[0]*a[3]

	q = np.array( [w,x,y,z] )

	return q
	
def quatRot(q_toBfromA, v_expA):
	"""
	Rotates vector by quaterion
	"""
	try:
		v_expA = np.hstack( (0,v_expA) )
	except:
		v_expA = np.vstack( (np.zeros(v_expA.shape[1]),v_expA) )
		
	#v_expB = quatMult(q_toBfromA, quatMult( v_expA, quatConj(q_toBfromA)))
	v_expB = quatMult( quatConj(q_toBfromA), quatMult( v_expA, q_toBfromA))
	
	v_expB = v_expB[1:]
	
	return v_expB

