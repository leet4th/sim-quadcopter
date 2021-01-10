import numpy as np
import matplotlib.pyplot as plt

RAD2DEG = 180./np.pi
DEG2RAD = 1/RAD2DEG

def generate_plots(output):

    time = output['time'][0,:]
    pos_L = output['pos_L']
    pos_B = output['pos_B']
    vel_L = output['vel_L']
    vel_B = output['vel_B']
    qToBfromL = output['qToBfromL']
    ypr = output['ypr']
    wb = output['wb']
    wm = output['wm']
    wmMin = output['wmMin']
    wmMax = output['wmMax']
    quat_err = output['quat_err']
    
    fig,ax = plt.subplots(2,2,sharex=True)
    ax[0,0].plot(time, pos_L[0,:], label='x') 
    ax[0,0].plot(time, pos_L[1,:], label='y') 
    ax[0,0].plot(time, pos_L[2,:], label='z')
    ax[0,0].legend()
    ax[0,0].grid()
    ax[0,0].set_ylabel('Pos L')
    ax[0,1].plot(time, pos_B[0,:], label='x') 
    ax[0,1].plot(time, pos_B[1,:], label='y') 
    ax[0,1].plot(time, pos_B[2,:], label='z')
    ax[0,1].legend()
    ax[0,1].grid()
    ax[0,1].set_ylabel('Pos B')  
    ax[1,0].plot(time, vel_L[0,:], label='x') 
    ax[1,0].plot(time, vel_L[1,:], label='y') 
    ax[1,0].plot(time, vel_L[2,:], label='z')
    ax[1,0].legend()
    ax[1,0].grid()
    ax[1,0].set_ylabel('Vel L')
    ax[1,1].plot(time, vel_B[0,:], label='x') 
    ax[1,1].plot(time, vel_B[1,:], label='y') 
    ax[1,1].plot(time, vel_B[2,:], label='z')
    ax[1,1].legend()
    ax[1,1].grid()
    ax[1,1].set_ylabel('Vel B')        
    
    
    
    fig,ax = plt.subplots(3,1,sharex=True)
    ax[0].plot(time, qToBfromL[0], label='qw')
    ax[0].plot(time, qToBfromL[1], label='qx')
    ax[0].plot(time, qToBfromL[2], label='qy')
    ax[0].plot(time, qToBfromL[3], label='qz')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('qToBfromL')  

    ax[1].plot(time, ypr[0]*RAD2DEG, label='yaw')
    ax[1].plot(time, ypr[1]*RAD2DEG, label='pitch')
    ax[1].plot(time, ypr[2]*RAD2DEG, label='roll')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Euler321 (deg)')

    ax[2].plot(time, wb[0]*RAD2DEG, label='wx')
    ax[2].plot(time, wb[1]*RAD2DEG, label='wy')
    ax[2].plot(time, wb[2]*RAD2DEG, label='wz')
    ax[2].legend()
    ax[2].grid()
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Body Rates')    
    
    fig,ax = plt.subplots(1,1,sharex=True)
    ax.plot(time, wm[0], label='wm1')
    ax.plot(time, wm[1], label='wm2')
    ax.plot(time, wm[2], label='wm3')
    ax.plot(time, wm[3], label='wm4')    
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Speed')    
    ax.set_title(f'wmMin = {int(wmMin)}, wmMax = {int(wmMax)}')
    
    fig,ax = plt.subplots(1,1,sharex=True)
    ax.plot(time, quat_err[0], label='qw')
    ax.plot(time, quat_err[1], label='qx')
    ax.plot(time, quat_err[2], label='qy')
    ax.plot(time, quat_err[3], label='qz')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('quat_err')      
    
    
    

    
   
    
    