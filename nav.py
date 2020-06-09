import numpy as np

class MadgwickAHRS:
	def __init__(self, sampleFreq=100.0, beta=0.5):
		self.beta = beta
		self.q0 = 1.0
		self.q1 = 0.0
		self.q2 = 0.0
		self.q3 = 0.0
		self.wb = np.zeros(3)
		self.invSampleFreq = 1.0 / sampleFreq
		
	def updateImu(self,gx,gy,gz,ax,ay,az):
		q0 = self.q0
		q1 = self.q1
		q2 = self.q2
		q3 = self.q3
		
		# Rate of change of quaternion from gyroscope
		qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
		qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
		qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
		qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);	
		
		# Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
		if(not((ax == 0.0) and (ay == 0.0) and (az == 0.0))):
			# Normalise accelerometer measurement
			recipNorm = self.invSqrt(ax * ax + ay * ay + az * az);
			ax *= recipNorm;
			ay *= recipNorm;
			az *= recipNorm;

			# Auxiliary variables to avoid repeated arithmetic
			_2q0 = 2.0 * q0;
			_2q1 = 2.0 * q1;
			_2q2 = 2.0 * q2;
			_2q3 = 2.0 * q3;
			_4q0 = 4.0 * q0;
			_4q1 = 4.0 * q1;
			_4q2 = 4.0 * q2;
			_8q1 = 8.0 * q1;
			_8q2 = 8.0 * q2;
			q0q0 = q0 * q0;
			q1q1 = q1 * q1;
			q2q2 = q2 * q2;
			q3q3 = q3 * q3;

			# Gradient decent algorithm corrective step
			s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
			s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
			s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
			s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay;
			recipNorm = self.invSqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3); # normalise step magnitude
			s0 *= recipNorm;
			s1 *= recipNorm;
			s2 *= recipNorm;
			s3 *= recipNorm;

			# Apply feedback step
			qDot0 -= self.beta * s0;
			qDot1 -= self.beta * s1;
			qDot2 -= self.beta * s2;
			qDot3 -= self.beta * s3;	

		# Integrate rate of change of quaternion to yield quaternion
		q0 += qDot0 * self.invSampleFreq;
		q1 += qDot1 * self.invSampleFreq;
		q2 += qDot2 * self.invSampleFreq;
		q3 += qDot3 * self.invSampleFreq;

		# Normalise quaternion
		recipNorm = self.invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
		q0 *= recipNorm;
		q1 *= recipNorm;
		q2 *= recipNorm;
		q3 *= recipNorm;
		
		self.q0 = q0
		self.q1 = q1
		self.q2 = q2
		self.q3 = q3
		self.wb = np.array([gx,gy,gz])
	
	def getBodyRate(self):
		return self.wb

	def getQuaternion(self):
		return np.array([self.q0, self.q1, self.q2, self.q3])
			
	def invSqrt(self, x):
		return 1/np.sqrt(x)
	