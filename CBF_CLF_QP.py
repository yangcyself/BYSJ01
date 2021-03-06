"""
    This file defines the CBF and the CBF-QP controller
"""
from problemSetting import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import copy
import json
from util.visulization import drawEclips


# class BF:
#     """
#         A simple Barrier Function(BF) for making the distance of the position to the origin is larger than _r
#             h =  x^2 + y^2 - r^2
#             B = dh + mc * h =  2 * x * vx + 2 * y * vy  +  mc ( x^2 + y^2 - r^2 )
#         `mc` means gamma
#     """
#     def __init__(self, r=1, mc = 1):
#         self.r2 = r**2
#         self.mc = mc
#         self.P = np.array([ [mc, 0, 1, 0],
#                             [0, mc, 0, 1],
#                             [1, 0, 0, 0],
#                             [0, 1, 0, 0],])

#     def __call__(self, state):
#         return np.array(state).T @ self.P @ np.array(state) - self.r2

#     def dt(self,state,u):
#         # x^T A^TP x + x^T PA x
#         # + 
#         state = np.array(state)
#         u = np.array(u)
#         # print("state.T @ Dyn_A.T:", state.T @ Dyn_A.T)
#         # print("state.T @ (Dyn_A.T @ self.P + self.P @ Dyn_A ) @ state: ",state.T @ (Dyn_A.T @ self.P + self.P @ Dyn_A ) @ state)
#         return ( state.T @ (Dyn_A.T @ self.P + self.P @ Dyn_A ) @ state + 
#                  u.T @ Dyn_B.T @ self.P @ state +  state.T @ self.P @ Dyn_B @ u)


class BF:
    """
        A simple Barrier Function(BF) for making the distance of the position to the origin is larger than _r
            h =  x^2 + y^2 - r^2
            B = dh + mc * h =  2 * x * vx + 2 * y * vy  +  mc ( x^2 + y^2 - r^2 )
        `mc` means gamma
    """
    def __init__(self, A,b,c):
        self.A = A
        self.b = b
        self.c = c

    def __call__(self, state):
        state = np.array(state) 
        # print("h:", state[:2].T @ self.A[:2,:2] @ state[:2] + self.c, 'B: ', state.T @ self.A @ state + self.b.T @ state + self.c)
        return state.T @ self.A @ state + self.b.T @ state + self.c

    def dt(self,state,u):
        # x^T A^TP x + x^T PA x
        # + 
        state = np.array(state)
        u = np.array(u)
        return (( state.T @ (Dyn_A.T @ self.A + self.A @ Dyn_A ) @ state + 
                 u.T @ Dyn_B.T @ self.A @ state +  state.T @ self.A @ Dyn_B @ u) +  ## xAx
                 self.b.T @ (self.A @ state + Dyn_B @ u) ) ## bx
        # return (( state.T @ ((sys_A-np.eye(4)).T @ self.A + self.A @ (sys_A-np.eye(4)) ) @ state + 
        #          u.T @ sys_B.T @ self.A @ state +  state.T @ self.A @ sys_B @ u) +  ## xAx
        #          self.b.T @ (self.A @ state + sys_B @ u) ) ## bx


class CBF(BF):
    """
        The Control barrier function based on BF, the main difference is that CBF's state is set, u is the only variable
    """
    def __init__(self, state, A, b, c, mc2 = 1):
        super().__init__(A,b,c)
        self.state = state
        self.mc2 = mc2

    def __call__(self,u):
        return super().dt(self.state,u) + self.mc2 * super().__call__(self.state) 
    
    def grad(self,u):
        return jacobian(self.__call__)(u)[:]
    
    def B(self):
        return super().__call__(self.state) 
    
    def dB(self,u):
        return super().dt(self.state,u)


class CLF:
    def __init__(self, state, t, mc2 = 1000):
        self.state = np.array(state).astype(np.float)
        self.mc2 = mc2

        # self.x_des = 2*t - 2.5
        # # self.y_des = np.sin(2*t-3)
        self.y_des = 0
        self.x_des = 2*t-3
        # self.x_des = 12*np.sin(t/4)-3
        # self.y_des = 2*np.cos(2*t)
        # self.x_des = 0
        # self.y_des = 2*t-2.5
        self.h = lambda x: (x[0]-self.x_des)**2 + (x[1]-self.y_des)**2


    def __call__(self,u):
        u = np.array(u)
        # LfLf_h(x) + LfLg_h(x)
        # print("Jac:", jacobian(self.h)(self.state).T @ Dyn_A)
        return jacobian(self.h)(self.state).T @ Dyn_A @ ( Dyn_A @ self.state + Dyn_B @ u) + self.mc2 * self.h(self.state)
    
    def grad(self,u):
        u = np.array(u)
        return jacobian(self.__call__)(u)[:]

    def get_des(self):
        return np.array([self.x_des,self.y_des,0,0])


def CBF_CLF_QP(state, CBFfun, CLFfun,badpoints=[]):
    """
        a simple QP posed to have a target velocity toward x direction
        input state
        CBFfun,CBFfun: state -> CBF
        return u
    """

    cbf = CBFfun(state)
    clf = CLFfun(state)

    Wregularizer = 0.001
    def objective(dyn_u):
        """
            |u| + eps
        """
        # dyn_u = np.array([20,-20])
        # print(list(dyn_u))
        # print("OBJ", Wregularizer * sqeuclidean(dyn_u) + max(0,clf(dyn_u)) **2)
        return Wregularizer * sqeuclidean(dyn_u) + max(0,clf(dyn_u))


    def obj_grad(dyn_u):
        # dyn_u = np.array([20,-20])
        # print(state)
        # print(list(dyn_u))
        # print(clf.x_des)
        # print("jacobian(objective)(dyn_u)",jacobian(objective)(dyn_u))
        # print("GRAD:", 2 * Wregularizer * dyn_u + 2 * clf.grad(dyn_u) * (clf(dyn_u)>0) )
        return 2 * Wregularizer * dyn_u + clf.grad(dyn_u) * (clf(dyn_u)>0) 


    constraints = {'type':'ineq','fun': cbf, "jac":cbf.grad }

    x0 = np.random.random(2)
    # x0 = np.ones(2)
    
    bounds = np.ones((2,2)) * np.array([[-1,1]]) * MAX_INPUT
    options = {"maxiter" : 500, "disp"    : True}
    res = minimize(objective, x0, bounds=bounds,options = options,jac=obj_grad,
                constraints=constraints, method =  'SLSQP') # 'trust-constr' , "SLSQP"
    # assert(cbf(res.x)>-1e-9)
    res_x = np.nan_to_num(res.x,0)
    if(not cbf(res_x)>-1e-9):
        badpoints.append(state)
    print(cbf(res_x))
    print(cbf.B(),cbf.dB(res_x))
    # print(res_x.clip(-MAX_INPUT,MAX_INPUT))
    return res_x.clip(-MAX_INPUT,MAX_INPUT),badpoints


def CBF_QP_simulation(initState, episode = 10, *cbfarg):
    """
    The function that calles the CBF_QP given a init condition
    """
    xs = []
    us = []
    xdes = []
    x = initState
    for i in range(episode):
        try:
            u,bad = CBF_CLF_QP(x, lambda s: CBF(s, *cbfarg), lambda s:CLF(s, i*dt) )
            us.append(u)
            x = sys_A @ x + sys_B @ u
            xs.append(x)
            xdes.append(CLF(np.zeros(4),i*dt).get_des())
            print("x",x, "t ", i*dt)
            print("u",u)
        except KeyboardInterrupt:
            break
        except AssertionError:
            print("ASSERTION ERROR")
            break

    traj = np.array(xs)
    plt.plot(traj[:,0],traj[:,1], ".", alpha = 1, markersize = 14,label = 'actual traj')
    N = 36
    for c in collisionList:
        plt.plot(c.x + np.sqrt(c.r2)*np.cos(np.arange(0,(2+1./N)*3.14,2*3.14/N)),
                 c.y + np.sqrt(c.r2)*np.sin(np.arange(0,(2+1./N)*3.14,2*3.14/N)), label = 'obstacle',c = "r")
    # print(us)
    xdes = np.array(xdes)
    ax = plt.gca()
    ax.plot(xdes[:,0],xdes[:,1],".",alpha = 0.8, label = 'command traj')
    if(len(bad)):
        bad = np.array(bad)
        ax.plot(bad[:,0],bad[:,1],"o", label = "Optimization Fail")
    plt.legend()
    # lim = max(np.max(traj[:,0])-np.min(traj[:,0]),6, 2*(np.max(traj[:,1])-np.min(traj[:,1])))
    lim = max(np.max(traj[:,1])-np.min(traj[:,1]),6, 2*(np.max(traj[:,0])-np.min(traj[:,0])))
    # plt.xlim((-3.3,-3.3+lim))
    # plt.ylim((-lim/2, lim/2))
    plt.ylim((-3.3,-3.3+lim))
    plt.xlim((-lim/2, lim/2))
    # for i in range(episode):
    return ax



if __name__ == "__main__":
    # Polyparameter = json.load(open("data/exp1/svmopt.json","r"))
    Polyparameter = json.load(open("data/exp1/svm_complete.json","r"))
    # Polyparameter = json.load(open("data/tmp/baseline.json","r"))
    # Polyparameter = json.load(open("data/exp1/svm_def_aug.json","r"))
    # Polyparameter = json.load(open("data/exp1/svm.json","r"))
    # Polyparameter = json.load(open("data/tmp/svm_def.json","r"))
    sign = np.sign(Polyparameter["A"][0][0])
    A,b,c = sign * np.array(Polyparameter["A"]), sign * np.array(Polyparameter["b"]), sign * np.array(Polyparameter["c"])
    
    # B = BF(A,b,c)
    # print(B([10.        ,  8.79382108, -8.30894843, -0.57724171]))
    # # print(B([1,1,0,0]))
    # # print(B([1,0,0,0]))
    # # print(B([1,1,-1,-1]))
    # print(B([-2.5,0,1,0]))

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

##### PLOT STYLE

    import matplotlib.pyplot as plt
    # import numpy as np
    import sys
    import matplotlib
    import seaborn as sns
    import matplotlib.gridspec as gspec

    matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rc('font',family='serif', serif=['Palatino'])
    matplotlib.rc('font',family='times', serif=['Palatino'])
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    sns.set_style('white')
    # plt.rcParams['figure.figsize'] = [10, 8]
    def set_style():
        sns.set(font='serif', font_scale=1.4)
        
    # Make the background white, and specify the
        # specific font family
        sns.set_style("white", {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            'axes.facecolor': 'white',
            'lines.markeredgewidth': 1})
        
        plt.rcParams.update({'font.size': 21})
        plt.rc('axes', titlesize=25)     # fontsize of the axes title
        plt.rc('axes', labelsize=23)  

    legendParam = {"frameon":True,"framealpha" : 1,"edgecolor":"0","fontsize":23}

    set_style()

#####

    plt.figure(figsize=(8,8))

    CBF_QP_simulation([-3,0,0,0],50, # episode
         A,b,c) # c
    drawEclips(A,b,c)
    plt.legend(**legendParam)
    plt.xlabel("$p_x$")
    plt.ylabel("$p_y$")
    plt.xlim((-3,3))
    plt.ylim((-3,3))
    plt.savefig("toySimulate.png",bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    # print(CBF_CLF_QP([ 7.73014651, 10.        , -2.1482564 , -6.27712458],
    #         lambda s:CBF(s,A,b,c), lambda s:CLF(s,0.7) ))

    # def func(p):
    #     p = np.array(p)
    #     return (p.T @ A @ p + b.T @ p + c)

    # import pickle as pkl
    # from glob import glob
    # import os
    # loaddir = "./data/exp1"
    # X = []
    # y = []
    # for f in glob(os.path.join(loaddir,"*.pkl")):
    #     print(f)
    #     data = pkl.load(open(f,"rb"))
    #     X += [s['testPoint'] for s in data]
    #     y += [ 1 if (s['fvalue']<0) else -1 for s in data]
    # posX = np.array([x for x,y in zip(X,y) if y ==1])
    # for x in posX:
    #     print(x)
    #     print(CBF_CLF_QP(x,
    #             lambda s:CBF(s,A,b,c), lambda s:CLF(s,0.6) ))



    # mc = 1
    # CBF_QP_simulation([-3,0.,0,0],500, # episode
    #     np.array([[mc,0,1,0],
    #               [0,mc,0,1],
    #               [1,0, 0,0 ],
    #               [0,1, 0,0 ]]), # A
    #    - np.array([0., 0., 0., 0.]), # b
    #    - 1 ) # c
    # plt.show()

    # c= CLF([-3,0,0,0],0.1)
    # print(c([2,0]))
    # print(c.grad(np.array([1,0])))


