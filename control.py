import numpy as np
from transform import quatMult, quatConj


class quadControl:
	def __init__(self,nav):
		self.cmd = np.zeros(3) # body moment
		self.nav = nav
		self.Kp  = np.array([1,1,1])*100
		self.Kd  = np.array([1,1,1])*200
		
	def getCmd(self):
		self.qErr = quatMult(quatConj(self.nav.getQuaternion()),np.array([1,0,0,0]))
		

		self.cmd[0] = self.Kp[0]*self.qErr[1] - self.Kd[0] * self.nav.wb[0]
		self.cmd[1] = self.Kp[1]*self.qErr[2] - self.Kd[1] * self.nav.wb[1]
		self.cmd[2] = self.Kp[2]*self.qErr[3] - self.Kd[2] * self.nav.wb[2]
		
		
		return self.cmd