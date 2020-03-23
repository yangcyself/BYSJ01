import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

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


def drawEclips(A,b,c,ax = None):
    if(ax is None):
        ax = plt.gca()

    A,b,c = np.array(A)[:2,:2],np.array(b)[:2],np.array(c)
    sign = np.sign(A[0][0]) # assume A is either pos definate or neg definate
    A,b,c = sign * A, sign * b, sign * c
    A_inv = np.linalg.inv(A)
    c = c - 1/4 * b.T @ A_inv @ b
    b = 1/2 * A_inv @ b
    R = sqrtm(A_inv)

    N = 36
    points = np.array([np.sqrt(-c) * R @ np.array([np.cos(th),np.sin(th)]) + b for th in np.arange(0,(2+1./N)*3.14,2*3.14/N)])
    ax.plot(points[:,0],points[:,1])