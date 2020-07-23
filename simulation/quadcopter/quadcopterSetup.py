import numpy as np

from transform import euler3212quat, quatRot

RAD2DEG = 180./np.pi
DEG2RAD = 1/RAD2DEG

def init_model():
    data = {}
    
    # Constants
    data['mass'] = 1.2  # kg
    data['g']    = 9.81 # m/s^2
    data['L']    = 0.16 # m
    data['Ixx']  = 0.0123
    data['Iyy']  = 0.0123
    data['Izz']  = 0.0123
    data['kF']   = 1.076e-5 # N/(rad/s)^2
    data['kM']   = 1.632e-7 # Nm/(rad/s)^2
    
    # Initial State vector
    state, cmd_hover = init_state(data)
    
    return data, state, cmd_hover
    
def init_state(data):

    wmHover = np.sqrt(data['mass']*data['g']/(4*data['kF']))
    
    # Yaw, Pitch, Roll
    euler321 = np.array([0,45,45], dtype='float') * DEG2RAD
    qToBfromL = euler3212quat(euler321)
    
    pos_L = np.array([0,0,0], dtype='float')
    vel_L = np.array([0,0,0], dtype='float')
    
    pos_B = quatRot(qToBfromL, pos_L)
    vel_B = quatRot(qToBfromL, vel_L)
    
    state = np.zeros(13)
    state[0:3]  = pos_B[:]
    state[3:7]  = qToBfromL[:]
    state[7:10] = vel_B[:]
    state[10]   = 0. # wbx
    state[11]   = 0. # wby
    state[12]   = 0 * DEG2RAD # wbz  
    
    return state, wmHover