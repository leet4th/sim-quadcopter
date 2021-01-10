import numpy as np
from transform import quatMult, quatRot, quatConj

RAD2DEG = 180./np.pi
DEG2RAD = 1/RAD2DEG

class QuadControl:
    """
    Class for quadcotper control algorithm
    """
    
    def __init__(self, dt, controlGains): 
    
        self.dt = dt
        
        # Init parameters
        self.gain_posL_P = np.zeros(3)

        # Set gains
        self.set_gains(controlGains)
        
        # State set points
        self.posL_sp = np.zeros(3)
        self.velL_sp = np.zeros(3)
        self.quat_sp = np.zeros(3)
        self.rate_sp = np.zeros(3)
        self.thrust_int = np.zeros(3)
        self.rate = np.zeros(3)
        self.cmd = np.zeros(4)
        self.wCmd = np.zeros(4)

        
    def set_gains(self, controlGains):
        # position loop
        self.gain_posL_P    = controlGains["gain_posL_P"]
        # velocity loop
        self.gain_velL_P    = controlGains["gain_velL_P"]
        self.gain_velL_I    = controlGains["gain_velL_I"]
        self.gain_velL_D    = controlGains["gain_velL_D"]
        self.gain_velL_zFF  = controlGains["gain_velL_zFF"]
        self.gain_velL_max  = controlGains["gain_velL_max"]
        # Attitude loop
        self.gain_att_P     = controlGains["gain_att_P"]
        # Rate loop
        self.gain_rate_P    = controlGains["gain_rate_P"]
        self.gain_rate_D    = controlGains["gain_rate_D"]
        # command to motor mixer
        self.mixerCmd2Motor = controlGains["mixerCmd2Motor"]
        self.wCmdMin        = controlGains["wCmdMin"]
        self.wCmdMax        = controlGains["wCmdMax"]
        
    def update(self, nav, des):
        """
        Processes nav data and desired state to generate motor commands. 
        This method serves as the main control loop
        """
    
        # Update command set points
        self.posL_sp[:] = des.posL_sp
    
        # Update control loops
        self.update_posL(nav)
        self.update_velL(nav)
        self.update_tilt(nav)
        self.update_attitude(nav)
        self.update_rate(nav)
        self.update_motor()
    
    def update_posL(self, nav):
        """
        Position control loop
        P controller for position in L frame
        
        Generates velL set point
        """
        # Calculate error
        posL_err = self.posL_sp - nav.pos_L
        
        # P controller to set velocity set point
        self.velL_sp = self.gain_posL_P * posL_err
        
        # Limit velocity set point to reduce motor saturation
        self.limit_velL_command()
        
    def limit_velL_command(self):
        """
        Limit the velocity set point reduce motor saturation
        """
        velL_sp_norm = np.linalg.norm(self.velL_sp)
        if velL_sp_norm > self.gain_velL_max:
            self.velL_sp *= self.gain_velL_max/velL_sp_norm
            
    def update_velL(self, nav):
        """
        Velocity control loop
        PID controller for velocity in L frame
        Includes feed forward for velL_z to account for gravity term
        
        Generates thrust set point
        
        TODO: Add anti windup logic to integrator
        """
        # Calculate error
        velL_err = self.velL_sp - nav.vel_L
        
        # PD controller
        self.thrust_sp = self.gain_velL_P * velL_err - self.gain_velL_D * nav.vel_L
        
        # I Controller
        self.thrust_sp += self.thrust_int
        
        # Feed Forward gravity term to z axis
        self.thrust_sp[2] -= self.gain_velL_zFF
        
        # Update integrator for next update
        self.thrust_int += self.gain_velL_I * velL_err * self.dt
        
    def update_tilt(self, nav):
        """
        Convert thrust set point to attitude controller set point
        
        Generates attitude quaternion set point
        """
        self.quat_sp = np.array([1,0,0,0],dtype='float')

    def update_attitude(self, nav):
        """
        Attitude control loop
        
        Generates body rate set point
        
        TODO: Add yaw control
        """
            
        self.quat_err = quatMult( self.quat_sp, nav.qToBfromL)
        
        if self.quat_err[0] < 0:
            self.quat_err = quatConj(self.quat_err)
        self.rate_sp = 2.0 * self.gain_att_P * self.quat_err[1:]
        
        #self.rate_sp = 2.0 * self.gain_att_P * np.sign(quat_err[0]) * quat_err[1:]
        
        pass

    def update_rate(self, nav):
        """
        Body rate control loop
        
        Generates motor commands
        """
        #self.rate_sp = np.array([0,0,0]) * DEG2RAD
        rate_err = self.rate_sp - nav.wb
        self.rate = self.gain_rate_P * rate_err - self.gain_rate_D * nav.wb

    def update_motor(self):
        """
        Apply mixer to for motor command
        """
        #import IPython; IPython.embed()
        
        # Assemble thrust, body rate commands
        self.cmd[0]  = np.linalg.norm(self.thrust_sp)
        self.cmd[1:] = self.rate
                
        #self.cmd = np.array([580**2,0,0,0])
        
        # Transform command to motors
        wCmdRaw= self.mixerCmd2Motor.dot(self.cmd)
        
        # Scale wCmd by min/max command [0-1]
        wCmdRaw = (wCmdRaw - self.wCmdMin)/(self.wCmdMax - self.wCmdMin)
        self.wCmd = np.clip(wCmdRaw,0,1) 
    
  

