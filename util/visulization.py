import matplotlib.pyplot as plt
import numpy as np

from problemSetting import *

def plotAction(initState, dyn_u):

    x = initState
    traj = []
    for u in dyn_u.reshape(-1,2):
        x = sys_A @ x + sys_B @ u
        traj.append(x)        
    traj = np.array(traj)
    plt.plot(traj[:,0],traj[:,1],"-","x")


    N = 36
    for c in collisionList:
        plt.plot(c.x + np.sqrt(c.r2)*np.cos(np.arange(0,(2+1./N)*3.14,2*3.14/N)),
                 c.y + np.sqrt(c.r2)*np.sin(np.arange(0,(2+1./N)*3.14,2*3.14/N)))
    plt.show()
