
import numpy as np
import time as tm
import cProfile
import matplotlib.pyplot as plt

from animation import trajectory_animation
from plot_sim import generate_plots

from quadcopter.quadcopterModel import Quadcopter
from navigation.BasicNavigation import BasicNavigation
from quadControl.quadControl import QuadControl

# Main sim loop
def sim(t,dt,quad):
    # Nav
    nav.update()
       
    # Trajectory
    posCmd = np.array([10,10,-10])
    
    # Control
    wCmd = 1.5*wCmd_hover
    ctrl.update(nav)
    #cmd = np.hstack(([0,0,-quad.data['mass']*quad.data['g']],ctrl.cmd))
    #cmd = np.hstack(([0,
    #                  0,
    #                  10*(-5 - nav.pos_L[2]) + 1*(0 - nav.vel_L[2]) - quad.data['mass']*quad.data['g'] ],
    #                  ctrl.cmd))
    #print(ctrl.cmd)
    
    # Update quadcopter model
    quad.update(t,dt,ctrl.wCmd)
   
# Setup time
tStart = 0.
dt = 0.005
tEnd = 500.
   
# Setup models
quad = Quadcopter(use_actuator_model=True)
wCmd_hover = np.array([quad.wmHover]*4)

nav = BasicNavigation(quad)

#                            T, Mx, My, Mz
mixerCmd2Motor = np.array([[ 1,  0,  1, -1], # Motor 1
                           [ 1, -1,  0,  1], # Motor 2
                           [ 1,  0, -1, -1], # Motor 3
                           [ 1,  1,  0,  1]  # Motor 4
                           ])
ctrl = QuadControl(mixerCmd2Motor)

# Setup Output dict structure
output = {}
output['time'] = []
output['pos_B'] = []
output['pos_L'] = []
output['qToBfromL'] = []
output['ypr'] = []
output['vel_B'] = []
output['vel_L'] = []
output['wb'] = []
output['wm'] = []
output['velDot_B'] = []
output['velDot_L'] = []
output['qDot'] = []
output['wbDot'] = []

# Run Sim
tic = tm.perf_counter()
tk = tStart
kPrint = 0
nPrint = 10
print('\nRunning sim...')
while tk <= tEnd:
    
    sim(tk, dt, quad)
    
    # Store output
    output['time'].append(tk)
    output['pos_B'].append(quad.pos_B)
    output['pos_L'].append(quad.pos_L)
    output['qToBfromL'].append(quad.qToBfromL)
    output['ypr'].append(quad.ypr)
    output['vel_B'].append(quad.vel_B)
    output['vel_L'].append(quad.vel_L)
    output['wb'].append(quad.wb)
    output['wm'].append(quad.wm)
    output['velDot_B'].append(quad.velDot_B)
    output['velDot_L'].append(quad.velDot_L)
    output['qDot'].append(quad.qDot)
    output['wbDot'].append(quad.wbDot)
    
    # print status
    if tk/tEnd > kPrint/nPrint:
        print(f'\t{int(tk/tEnd*100)}% complete')
        kPrint += 1
    
    tk += dt
    
# Process output for numpy arrays
for var in output.keys():
    output[var] = np.array(output[var]).T
    if len(output[var].shape) == 1:
        output[var] = np.expand_dims(output[var],axis=0)
print(f'\t100% complete')
toc = tm.perf_counter()
print(f"Completed in {toc-tic} seconds\n\n")

generate_plots(output)
#trajectory_animation(output, dt, type='point_mass')



plt.show()









