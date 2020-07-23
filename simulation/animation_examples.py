
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

RAD2DEG = 180 / np.pi
DEG2RAD = 1/RAD2DEG

def generate_pos_data(time):

    data = np.zeros((3,len(time)))
    
    for k,tk in enumerate(time):
        data[0,k] = (5+0.01*tk*tk)*np.cos(5 * DEG2RAD *tk)
        data[1,k] = (5+0.01*tk*tk)*np.sin(5 * DEG2RAD *tk)
        data[2,k] = 10 * tk - 0.075*tk*tk
            
    return data
    
time = np.linspace(0,130,500)
pos = generate_pos_data(time)      

x_lim = [min(pos[0,:]), max(pos[0,:])]
y_lim = [min(pos[1,:]), max(pos[1,:])]
z_lim = [min(pos[2,:]), max(pos[2,:])]

fig = plt.figure()
ax = p3.Axes3D(fig)
line1,     = ax.plot([], [], [], lw=3, color='red', marker='o')
line_traj, = ax.plot([], [], [], lw=3, color='black', linestyle='-')
line_traj_full, = ax.plot([], [], [], lw=1, color='green', linestyle='--')

ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_zlim(z_lim)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def update_lines(k):
   
    line1.set_data(pos[0,k], pos[1,k])
    line1.set_3d_properties(pos[2,k])
    
    line_traj.set_data(pos[0,:k], pos[1,:k])
    line_traj.set_3d_properties(pos[2,:k])
    
    line_traj_full.set_data(pos[0,:], pos[1,:])
    line_traj_full.set_3d_properties(pos[2,:])
       
    return line1, line_traj,line_traj_full
    
def init_plot():
    line1.set_data([],[])
    line1.set_3d_properties([])
    line_traj.set_data([],[])
    line_traj.set_3d_properties([])
    line_traj_full.set_data([],[])
    line_traj_full.set_3d_properties([])    
    return line1, line_traj, line_traj_full

line_ani = animation.FuncAnimation(fig, update_lines, init_func=init_plot,
                                       frames=len(time),interval = 5,
                                       blit=True)

plt.show()

