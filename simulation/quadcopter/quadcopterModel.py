import numpy as np
from scipy.integrate import ode

from quadcopter.quadcopterSetup import init_model
from transform import quatNormalize, quatRot, quatConj, quat2euler321

class Quadcopter:
    
    def __init__(self, tStart=0., use_actuator_model=True):
    
        # Option to use actuator model or direct mapping of cmd to 
        # body frame force and moments
        self.use_actuator_model = use_actuator_model
    
        # Initalize model parameters
        self.data, self.state, self.wmHover = init_model()
        
        # Initial states
        self.pos_B = self.state[0:3]
        self.qToBfromL = self.state[3:7]
        self.vel_B = self.state[7:10]
        self.wb = self.state[10:13]
        self.wm = np.array([self.wmHover]*4)
        
        # Initial dstate
        self.velDot_B = np.zeros(3)
        self.qDot = np.zeros(4)
        self.wbDot = np.zeros(3)
        self.wmDot = np.zeros(3)
        
        # Calculate other parameters
        self.calc_other()
        self.calc_force_moment()
        
        # Integrator
        self.integrator = ode(self.calc_dstate)
        self.integrator.set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, tStart)
        
    def calc_other(self):
        # Transform parameters expressed in B frame to L frame
        qToLfromB = quatConj(self.qToBfromL)
        self.pos_L = quatRot(qToLfromB, self.pos_B)
        self.vel_L = quatRot(qToLfromB, self.vel_B)
        self.velDot_L = quatRot(qToLfromB, self.velDot_B)
        self.ypr = quat2euler321(self.qToBfromL)
        
            
    def calc_force_moment(self):
        self.mThrust = self.data['kF']*self.wm#*self.wm
        self.mTorque = self.data['kM']*self.wm#*self.wm
        
        
        
    def calc_dstate(self, t, state, cmd):
        
        m = self.data['mass']
        g = self.data['g']
        L = self.data['L']
        Ixx = self.data['Ixx']
        Iyy = self.data['Iyy']
        Izz = self.data['Izz']
        
        # Unpack state
        x   = state[0]  # body frame
        y   = state[1]  # body frame
        z   = state[2]  # body frame
        qw  = state[3]  # to body from L
        qx  = state[4]  # to body from L
        qy  = state[5]  # to body from L
        qz  = state[6]  # to body from L
        vx  = state[7]  # body frame
        vy  = state[8]  # body frame
        vz  = state[9]  # body frame
        wx  = state[10] # body frame
        wy  = state[11] # body frame
        wz  = state[12] # body frame
        
        if self.use_actuator_model:
            # Calculate motor force and moment
            self.wm = cmd
            self.calc_force_moment()
            F1, F2, F3, F4 = self.mThrust
            M1, M2, M3, M4 = self.mTorque
            
            # Sum of forces in body frame
            Fx = 0.
            Fy = 0.
            Fz = -(self.mThrust[0]+self.mThrust[1]+self.mThrust[2]+self.mThrust[3])
            # Sum of moments in body frame
            Mx = self.data['L']*(-self.mThrust[1] + self.mThrust[3])
            My = self.data['L']*(self.mThrust[0] - self.mThrust[2])
            Mz = -self.mTorque[0] + self.mTorque[1] - self.mTorque[2] + self.mTorque[3] 
            
            """       
            # Calc dstate
            dstate = np.array([
                        vx,
                        vy,
                        vz,
                        -0.5*qx*wx - 0.5*qy*wy - 0.5*qz*wz,
                        0.5*qw*wx + 0.5*qy*wz - 0.5*qz*wy,
                        0.5*qw*wy - 0.5*qx*wz + 0.5*qz*wx,
                        0.5*qw*wz + 0.5*qx*wy - 0.5*qy*wx,
                        -2*g*qw*qy + 2*g*qx*qz + vy*wz - vz*wy,
                        2*g*qw*qx + 2*g*qy*qz - vx*wz + vz*wx,
                        g*qw**2 - g*qx**2 - g*qy**2 + g*qz**2 + vx*wy - vy*wx + (-F1 - F2 - F3 - F4)/m,
                        (Iyy*wy*wz - Izz*wy*wz + L*(-F2 + F4))/Ixx,
                        (-Ixx*wx*wz + Izz*wx*wz + L*(F1 - F3))/Iyy,
                        (Ixx*wx*wy - Iyy*wx*wy - M1 + M2 - M3 + M4)/Izz
                    ])
            """
            # Calc dstate
            dstate = np.array([
                        vx,
                        vy,
                        vz,
                        -0.5*qx*wx - 0.5*qy*wy - 0.5*qz*wz,
                        0.5*qw*wx + 0.5*qy*wz - 0.5*qz*wy,
                        0.5*qw*wy - 0.5*qx*wz + 0.5*qz*wx,
                        0.5*qw*wz + 0.5*qx*wy - 0.5*qy*wx,
                        Fx/m - 2*g*qw*qy + 2*g*qx*qz + vy*wz - vz*wy,
                        Fy/m + 2*g*qw*qx + 2*g*qy*qz - vx*wz + vz*wx,
                        Fz/m + g*qw**2 - g*qx**2 - g*qy**2 + g*qz**2 + vx*wy - vy*wx,
                        (Iyy*wy*wz - Izz*wy*wz + Mx)/Ixx,
                        (-Ixx*wx*wz + Izz*wx*wz + My)/Iyy,
                        (Ixx*wx*wy - Iyy*wx*wy + Mz)/Izz
                    ])                          
        else:
            # Using direct mapping of cmd to body frame force and moments
            Fx, Fy, Fz, Mx, My, Mz = cmd
            
            # Calc dstate
            dstate = np.array([
                        vx,
                        vy,
                        vz,
                        -0.5*qx*wx - 0.5*qy*wy - 0.5*qz*wz,
                        0.5*qw*wx + 0.5*qy*wz - 0.5*qz*wy,
                        0.5*qw*wy - 0.5*qx*wz + 0.5*qz*wx,
                        0.5*qw*wz + 0.5*qx*wy - 0.5*qy*wx,
                        Fx/m - 2*g*qw*qy + 2*g*qx*qz + vy*wz - vz*wy,
                        Fy/m + 2*g*qw*qx + 2*g*qy*qz - vx*wz + vz*wx,
                        Fz/m + g*qw**2 - g*qx**2 - g*qy**2 + g*qz**2 + vx*wy - vy*wx,
                        (Iyy*wy*wz - Izz*wy*wz + Mx)/Ixx,
                        (-Ixx*wx*wz + Izz*wx*wz + My)/Iyy,
                        (Ixx*wx*wy - Iyy*wx*wy + Mz)/Izz
                    ])                    
        
        # Store sum of forces and moments
        self.Fb = np.array([Fx,Fy,Fz])
        self.Mb = np.array([Mx,My,Mz])
        
        # Store dstate vector before returning for integrator
        self.dstate = dstate

        return dstate
        
    def update(self, t, dt, cmd):
        
        self.integrator.set_f_params(cmd)
        self.state = self.integrator.integrate(t,t+dt)     
        self.state[3:7] = quatNormalize(self.state[3:7])
   
        # Unpack state vector
        self.pos_B = self.state[0:3]
        self.qToBfromL = self.state[3:7]
        self.vel_B = self.state[7:10]
        self.wb = self.state[10:13]
        
        # Unpack dstate vector
        self.qDot = self.dstate[3:7]
        self.velDot_B = self.dstate[7:10]
        self.wbDot = self.dstate[10:13]
        
        self.calc_other()

        

        