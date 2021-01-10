
import numpy as np
import time as tm
import matplotlib.pyplot as plt

from animation import trajectory_animation
from plot_sim import generate_plots

from quadcopter.quadcopterModel import Quadcopter
from navigation.BasicNavigation import BasicNavigation
from quadControl.quadControl import QuadControl
from quadControl.quadGains import controlGains

class SingleWaypoint:
    def __init__(self, waypoint_posL=np.zeros(3)):
        self.posL_sp = waypoint_posL

   
# Setup time
tStart = 0.
dt = 0.005
tEnd = 100
   
# Setup models
quad = Quadcopter(use_actuator_model=True)
wCmd_hover = np.array([quad.wmHover]*4)

des = SingleWaypoint()

nav = BasicNavigation(quad)

dt_control = 0.01 # sec                           
ctrl = QuadControl(dt_control, controlGains)

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
output['quat_err'] = []

# Main sim loop
tControlPrev = -999.0
def sim(t, quad, dt_sim, dt_control):
    global tControlPrev

    # Nav
    nav.update()
    
    # Des
    if t > 5:
        des.posL_sp[2] = -100
    
    # Control
    if (t - tControlPrev) > dt_control:
        ctrl.update(nav, des)
        tControlPrev = t
        
    #ctrl.wCmd = np.zeros(4)
    # Update quadcopter model
    #ctrl.wCmd = np.array([1,1,1,1])*0.7
    quad.update(t,dt_sim,ctrl.wCmd)

# Run Sim
tic = tm.perf_counter()
tk = tStart
kPrint = 0
nPrint = 10
print('\nRunning sim...')
while tk <= tEnd:
    
    sim(tk, quad, dt, dt_control)
    
    # Store output
    #sim
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
    #control
    output['quat_err'].append(ctrl.quat_err)
    
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
# Save key constants
output['wmMin'] = quad.wmMin
output['wmMax'] = quad.wmMax
        
        
print(f'\t100% complete')
toc = tm.perf_counter()
print(f"Completed in {toc-tic} seconds\n\n")

generate_plots(output)
trajectory_animation(output, dt, type='point_mass')



plt.show()









