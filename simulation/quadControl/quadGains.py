import numpy as np

# Control Gains dict
controlGains = {}
# position loop
controlGains["gain_posL_P"] = np.array([1, 1, 1]) * 1 
# velocity loop
controlGains["gain_velL_P"] = np.array([1, 1, 1]) * -50
controlGains["gain_velL_I"] = np.array([1, 1, 1]) * -15 * 0
controlGains["gain_velL_D"] = np.array([1, 1, 1]) * -1 
controlGains["gain_velL_zFF"] = -1.2*9.81  *  44.23 
controlGains["gain_velL_max"] = 5 
# Attitude loop
controlGains["gain_att_P"] = np.array([1, 1, 1]) * -10
# Rate loop
controlGains["gain_rate_P"] = np.array([1, 1, 1])*10
controlGains["gain_rate_D"] = np.array([1, 1, 1])*0.1
# Command to motor mixer
#                                            T, Mx, My, Mz
controlGains["mixerCmd2Motor"] = np.array([[ 1,  0,  1, -1], # Motor 1
                                           [ 1, -1,  0,  1], # Motor 2
                                           [ 1,  0, -1, -1], # Motor 3
                                           [ 1,  1,  0,  1]  # Motor 4
                                           ])
controlGains["wCmdMin"] = 0.2*523                                           
controlGains["wCmdMax"] = 1.5*523                                           