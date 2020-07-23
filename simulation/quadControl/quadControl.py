import numpy as np
from transform import quatMult, quatRot, quatConj

class QuadControl:
    def __init__(self, mixerCmd2Motor):
        self.Kp = np.array([ 4500.,
                             1000,
                             1000])
        self.Kd = np.array([ 4000.,
                             1000,
                             1000])                               
        self.mixerCmd2Motor = mixerCmd2Motor
               
    def update(self, nav):

        # Attitude and rate commands
        qCmd = np.array([1,0,0,0],dtype='float')
        wCmd = np.array([0,0,0],dtype='float')
        pos_L_cmd = np.array([0,0,0],dtype='float')
        
        # Feedback
        qNav = nav.qToBfromL
        wNav = nav.wb
        pos_L = nav.pos_L
        vel_L = nav.vel_L
        
        # Position loop
        posErr = pos_L_cmd - pos_L
        cmdPos = 10*posErr[2] + (523*523) + 5*(0- vel_L[2])
        
        # Attitude loop
        self.qErr = quatMult(qCmd, qNav)
        if self.qErr[0] < 0:
            self.qErr = quatConj(self.qErr)
        self.cmdMoment = -self.Kp*self.qErr[1:] - self.Kd*wNav
        self.cmd = np.hstack((cmdPos,self.cmdMoment))
        
        # Transform from command frame to motor frame
        #self.cmdClipped = np.clip(self.cmd,None,None)
        
        
        #self.wCmd = self.mixerCmd2Motor.dot(self.cmdClipped)
        self.wCmd = self.mixerCmd2Motor.dot(self.cmd)
        