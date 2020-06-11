import numpy as np
from sim.transform import quatRot, skew, dQuat, quatConj, quatNormalize
import scipy.integrate as integrate

class rigidBodyDynamics:
	def __init__(self, pos_L, vel_B, qToBfromL, wb, mass, inertia, gravity):
		# Constants
		self.mass      = mass		
		self.J         = inertia
		self.invJ      = np.linalg.inv(inertia)
		self.gravity   = np.array([0,0,gravity])
		
		# State vector
		self.state        = np.zeros(13)
		self.state[0:3]   = pos_L
		self.state[3:6]   = vel_B
		self.state[6:10]  = quatNormalize(qToBfromL)
		self.state[10:13] = wb
		
		# State Derivative vector
		self.dstate = np.zeros(13)
		
		# Initialize parameters
		self.Fb           = np.zeros(3)
		self.Mb           = np.zeros(3)
		
		# Build output dict
		self.output={}
		self.output['pos_L'] = []
		self.output['vel_B'] = []
		self.output['qToBfromL'] = []
		self.output['wb'] = []
		self.output['vDot_B'] = []
		self.output['qDotToBfromL'] = []
		self.output['wbDot'] = []
		self.output['Fb'] = []
		self.output['Mb'] = []

		# Store Initial states and dstates
		#self.storeOutput()

	def calc(self, state, t, Fb, Mb):
		pos_L       = np.array( state[ 0:3 ] )
		vel_B       = np.array( state[ 3:6 ] )
		qToBfromL   = np.array( state[ 6:10] )
		wb          = np.array( state[10:13] )

		wb_skew = skew(wb)
		
		# Linear equation of motion
		gravity_B = quatRot(qToBfromL, self.gravity)
		vDot_B    = Fb/self.mass + gravity_B - np.matmul(wb_skew, vel_B)
		
		# Rotational equations of motion
		J_wbDot = Mb - np.matmul( wb_skew, np.matmul(self.J, wb) )
		wbDot = np.matmul(self.invJ, J_wbDot)
		
		# Rotational kinematic
		qDotToBfromL = dQuat(qToBfromL, wb)
		
		# Pack dstate
		dstate = np.zeros(13)
		dstate[ 0:3 ] = quatRot(quatConj(qToBfromL), vel_B)
		dstate[ 3:6 ] = vDot_B
		dstate[ 6:10] = qDotToBfromL
		dstate[10:13] = wbDot
		
		return dstate

	def update(self, dt, getBodyForceMoment):
		self.Fb, self.Mb = getBodyForceMoment()		
		self.dstate = self.calc( self.state, dt, self.Fb, self.Mb )
		self.state = integrate.odeint( self.calc, self.state, [0,dt], args=(self.Fb, self.Mb) )[1]
		
		# Normalize quaternion after integration
		self.state[6:10] = quatNormalize( self.state[6:10] )
		
		# Save current state and dstate vectors to output dict
		self.storeOutput()
		
	def getOutput(self):
		output = {}
		for var in self.output.keys():
			output[var] = np.array(self.output[var]).T
		return output		
		
	def storeOutput(self):
		
		# State vector
		self.output['pos_L'].append(np.array( self.state[0:3] ))
		self.output['vel_B'].append(np.array( self.state[3:6] ))
		self.output['qToBfromL'].append(np.array( self.state[6:10] ))
		self.output['wb'].append(np.array( self.state[10:13] ))
		
		# State Derivative vector
		self.output['vDot_B'].append(np.array( self.state[3:6] ))
		self.output['qDotToBfromL'].append(np.array( self.state[6:10] ))
		self.output['wbDot'].append(np.array( self.state[10:13] ))
		
		# Force and Moment
		self.output['Fb'].append(self.Fb)
		self.output['Mb'].append(self.Mb)

class motor:
	def __init__(self, w, spin_dir, force_coef, torque_coef, tau, loc, vec):
		# Constants
		self.spin_dir = spin_dir
		self.force_coef = force_coef
		self.torque_coef = torque_coef
		self.tau = tau
		self.loc = loc
		self.vec = vec
		
		# State vector		
		self.state = np.zeros(1)
		self.state[0] = w
		
		# State Derivative vector		
		self.dstate = np.zeros(1)
		
		# Initialize parameters		
		self.calcForceTorque()
		
		# Build output dict
		self.output={}
		self.output['w'] = []
		self.output['dw'] = []
		self.output['force'] = []
		self.output['torque'] = []
		
		# Store Initial states and dstates
		#self.storeOutput()		
		
	def calc(self,state,t,wc):
		'''
		First order model with time constant
		'''
		
		w = state[0]
		dw = (wc - w)/self.tau
		dstate = [ dw ]
		return dstate
		
	def update(self, dt, wc):
		self.wc = wc
		self.dstate = self.calc( self.state, dt, wc )
		self.state = integrate.odeint( self.calc, self.state, [0,dt], args=(wc,) )[1]
		
		# Calculate motor force and torque from speed
		self.calcForceTorque()
		
		# Save current state and dstate vectors to output dict
		self.storeOutput()
	
	def calcForceTorque(self):
		# Calculate motor force and torque from speed
		self.force  = self.force_coef  * self.state[0]**2
		self.torque = self.spin_dir * self.torque_coef * self.state[0]**2		

	def getBodyForceMoment(self):
		force  = self.force * self.vec
		moment = self.torque * self.vec 
		return force, moment
		
	def getLocVec(self):
		return self.loc, self.vec

	def getOutput(self):
		output = {}
		for var in self.output.keys():
			output[var] = np.array(self.output[var]).T
		return output			
	
	def storeOutput(self):
		self.output['w'].append(self.state[0])
		self.output['dw'].append(self.dstate[0])
		self.output['force'].append(self.force)
		self.output['torque'].append(self.torque)

class motor2body:

	def __init__(self, motorList):
		self.motorList = motorList

	def getBodyForceMoment(self):
		Fb = np.zeros(3)
		Mb = np.zeros(3)
		for motor_k in self.motorList:
			motor_force, motor_torque = motor_k.getBodyForceMoment()
			motor_loc,motor_vec = motor_k.getLocVec()
			
			Fb += motor_force
			Mb += np.cross(motor_loc, motor_force)
			Mb += motor_torque
		return Fb, Mb