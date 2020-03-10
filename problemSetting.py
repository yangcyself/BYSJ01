"""
Environment:
    moving points collision avoidance
    State:  x,y,vx,vy
    Input: ax,ay
    constraint:
        Do not collide into bad area
        limited ax, and ay
"""

import numpy as np
import scipy.linalg as linalg

dt = 0.1

Dyn_A = np.array([[0,0,1,0],
                  [0,0,0,1],
                  [0,0,0,0],
                  [0,0,0,0]])

Dyn_B = np.array([[0,0],
                  [0,0],
                  [0.1,0],
                  [0,0.1]])

# change to distrete time system
sysdt = linalg.expm(np.concatenate([np.concatenate([Dyn_A,Dyn_B],axis = 1) * dt, np.zeros((2,6))],axis = 0))
sys_A = sysdt[:4,:4]
sys_B = sysdt[:4,-2:]


class collisionConstraint:
    def __init__(self,x,y,r):
        self.x = x
        self.y = y
        self.r2 = r**2
    
    def __call__(self,State):
        # print((State[0] - self.x)**2 + (State[1] - self.y)**2 - self.r2)
        return (State[0] - self.x)**2 + (State[1] - self.y)**2 - self.r2

collisionList = [
    collisionConstraint(0,0,1)
]


if __name__ == "__main__":
    print("sys_A",sys_A)
    print("sys_B",sys_B)