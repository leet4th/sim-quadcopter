import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

nFrames = 8

def trajectory_animation(data, dt, type='point_mass'):
    
    # Extract key data
    time = data['time'][0,:]
    pos  = data['pos_L']
    
    # Limits
    sideBuffer = 0.5
    limRange = 0.5 * max([pos[k,:].max() - pos[k,:].min() for k in range(3)]) + sideBuffer
    
    lim = []
    for k in range(3):
        midPoint = 0.5 * (pos[k,:].max() + pos[k,:].min())
        lim.append( [midPoint - limRange, midPoint + limRange] )
    
    # Setup figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    line_body  = ax.plot([], [], [], color='red', marker='o')[0]
    line_trail = ax.plot([], [], [], color='green', linestyle='-') [0]
    line_traj  = ax.plot([], [], [], color='black', linestyle='--') [0]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[1])
    ax.set_zlim(lim[2])

    def update_lines(k):
    
        pos_k = pos[:,k*nFrames]
        
        # body
        line_body.set_data(pos_k[0], pos_k[1])
        line_body.set_3d_properties(pos_k[2])
        # trail
        line_trail.set_data(pos[0,:k], pos[1,:k])
        line_trail.set_3d_properties(pos[2,:k])
        # traj
        line_traj.set_data(pos[0,:], pos[1,:])
        line_traj.set_3d_properties(pos[2,:])
        
        return line_body, line_trail, line_traj

    def init_plot():
        # body
        line_body.set_data([], [])
        line_body.set_3d_properties([])
        # trail
        line_trail.set_data([], [])
        line_trail.set_3d_properties([])
        # traj
        line_traj.set_data([], [])
        line_traj.set_3d_properties([])

        return line_body, line_trail, line_traj
    
    
    traj3D_ani = animation.FuncAnimation(fig,update_lines, init_func=init_plot,
                                            frames=len(time[0:-2:nFrames]), interval=(dt*1000*nFrames),
                                            blit=True)
    return traj3D_ani