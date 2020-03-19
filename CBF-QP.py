"""
    This file defines the CBF and the CBF-QP controller
"""
from problemSetting import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import copy



class BF:
    """
        A simple Barrier Function(BF) for making the distance of the position to the origin is larger than _r
            h =  x^2 + y^2 - r^2
            B = dh + mc * h =  2 * x * vx + 2 * y * vy  +  mc ( x^2 + y^2 - r^2 )
        `mc` means gamma
    """
    def __init__(self, r=1, mc = 1):
        self.r2 = r**2
        self.mc = mc
        self.P = np.array([ [mc, 0, 1, 0],
                            [0, mc, 0, 1],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],])

    def __call__(self, state):
        return np.array(state).T @ self.P @ np.array(state) - self.r2

    def dt(self,state,u):
        # x^T A^TP x + x^T PA x
        # + 
        state = np.array(state)
        u = np.array(u)
        # print("state.T @ Dyn_A.T:", state.T @ Dyn_A.T)
        # print("state.T @ (Dyn_A.T @ self.P + self.P @ Dyn_A ) @ state: ",state.T @ (Dyn_A.T @ self.P + self.P @ Dyn_A ) @ state)
        return ( state.T @ (Dyn_A.T @ self.P + self.P @ Dyn_A ) @ state + 
                 u.T @ Dyn_B.T @ self.P @ state +  state.T @ self.P @ Dyn_B @ u)


class CBF(BF):
    """
        The Control barrier function based on BF, the main difference is that CBF's state is set, u is the only variable
    """
    def __init__(self, state ,r=1, mc = 1, mc2 = 1):
        super().__init__(r,mc)
        self.state = state
        self.mc2 = mc2

    def __call__(self,u):
        return super().dt(self.state,u) + self.mc2 * super().__call__(self.state) 
    
    def grad(self,u):
        return jacobian(self.__call__)(u)[:]


def CBF_QP(state):
    """
        a simple QP posed to have a target velocity toward x direction
        input state
        return u
    """

    def objective(dyn_u):
        """
            L2 of the u and the v and v target
        """
        vtarget = np.array([2,0])
        # print(dyn_u,"new v:", sys_A[2:] @ state + sys_B[2:] @ dyn_u)
        return sqeuclidean((sys_A[2:,:] @ state + sys_B[2:] @ dyn_u) - vtarget ) + 0.00001 * sqeuclidean(dyn_u)


    def obj_grad(dyn_u):
        # print("jacobian(objective)(dyn_u)",jacobian(objective)(dyn_u))
        return jacobian(objective)(dyn_u)[:]

    constraints = {'type':'ineq','fun': CBF(state), "jac":CBF(state).grad }

    x0 = np.random.random(2)
    
    bounds = np.ones((2,2)) * np.array([[-1,1]]) * MAX_INPUT
    options = {"maxiter" : 500, "disp"    : False}
    res = minimize(objective, x0, bounds=bounds,options = options,jac=obj_grad,
                constraints=constraints)
    return res.x


def CBF_QP_simulation(initState, episode = 50):
    """
    The function that calles the CBF_QP given a init condition
    """
    xs = []
    us = []
    x = initState
    for i in range(episode):
        u = CBF_QP(x)
        us.append(u)
        x = sys_A @ x + sys_B @ u
        xs.append(x)
    
    traj = np.array(xs)
    plt.plot(traj[:,0],traj[:,1],".")
    N = 36
    for c in collisionList:
        plt.plot(c.x + np.sqrt(c.r2)*np.cos(np.arange(0,(2+1./N)*3.14,2*3.14/N)),
                 c.y + np.sqrt(c.r2)*np.sin(np.arange(0,(2+1./N)*3.14,2*3.14/N)))
    print(us)
    plt.show()



if __name__ == "__main__":
    # B = BF()
    # print(B([1,1,0,0]))
    # print(B([1,0,0,0]))
    # print(B([1,1,-1,-1]))
    # print(B([1,1,-0.2,-0.2]))

    # print("Test BF dt")
    # print(B.dt([1,1,0,0],[-1,-1]))
    # print(B.dt([1,1,0,0],[1,1]))

    # print("Test CBF")

    # B = CBF([1,1,0,0])
    # print(B(np.array([-1,-1])))
    # B = CBF([1,1,0,0])
    # print(B(np.array([1,1])))
    # B = CBF([0.5,0.5,0,0])
    # print(B(np.array([-1,-1])))

    # print(CBF_QP([-3,0,0,0]))

    CBF_QP_simulation([-3,0.,0,0],episode=50)