"""
Environment:
    moving points collision avoidance
    State:  x,y,vx,vy
    Input: ax,ay
    constraint:
        Do not collide into bad area
        limited ax, and ay
"""

# import numpy as np
import autograd.numpy as np
import scipy.linalg as linalg
from autograd import grad, jacobian

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
        # return (State[0] - self.x)**2 + (State[1] - self.y)**2 - self.r2
        return - np.log(np.sqrt((State[0] - self.x)**2 + (State[1] - self.y)**2)/ self.r2)
        # return - np.log(np.sqrt((State[0] - self.x)**2 / self.r2)) - np.log(np.sqrt((State[1] - self.y)**2 / self.r2))

    def grad(self,State):
        # res = np.zeros_like(State)
        # res[0] = 2*(State[0]-self.x)
        # res[1] = 2*(State[1]-self.y)
        
        ## using auto gard
        res = jacobian(self.__call__)(State)[:]
        return res

collisionList = [
    collisionConstraint(0,0,1)
]


if __name__ == "__main__":
    print("sys_A",sys_A)
    print("sys_B",sys_B)
    print(collisionList[0](np.array([0.0001,0.1, 0,1.8])))
    print(collisionList[0].grad(np.array([0.0001,0.1, 0,1.8])))
    print(collisionList[0](np.array([0.5,0.1, 0.5,1])))
    print(collisionList[0].grad(np.array([0.5,0.1, 0.5,1])))
    print(collisionList[0](np.array([0.1,1, 0,1])))
    print(collisionList[0].grad(np.array([0.1,1, 0,1])))
    print(collisionList[0](np.array([2,1,   1.3,1])))
    print(collisionList[0].grad(np.array([2,1,   1.3,1])))
    print(collisionList[0](np.array([1,0.1, 0.5,1])))
    print(collisionList[0].grad(np.array([1,0.1, 0.5,1])))
    print(collisionList[0](np.array([0.1,1, 0,1])))
    print(collisionList[0].grad(np.array([0.1,1, 0,1])))
    print(collisionList[0](np.array([2,1,   1.3,1])))
    print(collisionList[0].grad(np.array([2,1,   1.3,1])))