"""
    The pipeline for sample is:
        randomly select points(states)
        run optimization for each point
"""
from problemSetting import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import copy
from util.visulization import plotAction
from util.datalogger import DataLogger
import os
import pickle as pkl

HORIZON = 20
THRESHOULD = 0
# logger: this file takes a `logger` in environment

def optimize(initState, horizon = HORIZON):
    """
        Get a initstate and return the optimal constraint violation
            This serves as the sample function(f) in paper
    """
    initState = np.array(initState).astype(np.double)
    # print('\n initState:',initState)
    def constraintOftTraj(c):
        def returnfunc(dyn_u):
            result = np.zeros(len(dyn_u)//2)
            x = initState
            for i in range(len(dyn_u)//2):
                result[i] = c(x)
                x = sys_A @ x + sys_B @ dyn_u[2*i:2*i+2]
                # print(x)
            return result
        return returnfunc
    

    def jacOfTraj(c):
        def returnfunc(dyn_u):
            result = np.zeros((len(dyn_u)//2,len(dyn_u)))
            x = initState
            stateJac = np.zeros((4,len(dyn_u)))
            for i in range(len(dyn_u)//2):
                # result[i] = c(x)
                # print("StateJac%d:"%i,stateJac)
                # print("c grad:", c.grad(x).T)
                result[i,:] = c.grad(x).T @ stateJac
                x = sys_A @ x + sys_B @ dyn_u[2*i:2*i+2]
                stateJac = sys_A @ stateJac
                stateJac[:,2*i:2*i+2] = sys_B
            # print("constraint Jacobian",str(result))
            return result
        return returnfunc


    def objective(dyn_u):
        # return dyn_u .T @ dyn_u
        # print(-np.min([ np.min(constraintOftTraj(c)(dyn_u)) for c in collisionList]))
        # print("argmax", np.argmax(constraintOftTraj(collisionList[0])(dyn_u)))
        # print(constraintOftTraj(collisionList[0])(dyn_u))
        return np.max([ np.max(constraintOftTraj(c)(dyn_u)) for c in collisionList])


    def obj_grad(dyn_u):
        i = np.argmax([ np.max(constraintOftTraj(c)(dyn_u)) for c in collisionList])
        j = np.argmax(constraintOftTraj(collisionList[i])(dyn_u))
        return jacOfTraj(collisionList[i])(dyn_u)[j,:]

    # constraints = [{'type':'ineq','fun': constraintOftTraj(c), "jac":jacOfTraj(c) } for c in collisionList]

    # x0 = np.zeros(2*horizon)
    # x0 = np.ones(2*horizon)
    x0_whole = np.random.random(2*horizon)
    sol = np.array([])
    constraintViolation = 0
    for h in range(1,horizon):
        # gradually increase the horizon
        x0 = x0_whole[:2*h]
        x0[:len(sol)] = sol
        bounds = np.ones((2*h,2)) * np.array([[-1,1]]) * MAX_INPUT
        options = {"maxiter" : 500, "disp"    : False}
        res = minimize(objective, x0, bounds=bounds,options = options,jac=obj_grad)
                # constraints=constraints)

    # constraintViolation = np.linalg.norm(np.clip([c['fun'](res.x) for c in constraints],None,0)) 
        # print('\n initState:',initState)
        # print("solution:",res.x)
        constraintViolation = objective(res.x)
        # print("constraint violation:", constraintViolation)
        # plotAction(initState,res.x)
    
    return constraintViolation


class GaussionProcess:
    """
        This class maintains the states of gaussion process
            with the mu and sigma as described in eq 1 and eq 2 in the paper
    """
    def __init__(self, kernel, sigma2 = 0):
        self.kernel = kernel
        self.sigma2 = sigma2

        self.K = np.array([[]]) # the kernel matrix of already observed points
        self.kfunlist = []
        self.k = lambda x:np.array([[f(x)] for f in self.kfunlist]) # the function that returns the vector of a point ot observed points
        self.y = []


    def __call__(self,x):
        # return the posterior mu and sigma inferred from current gaussian process
        ktx = self.k(x)
        mu = ktx.T @ np.linalg.inv(self.K+self.sigma2*np.eye(len(self.y))) @ np.array(self.y)
        sigma = self.kernel(x,x) - ktx.T @ np.linalg.inv(self.K+self.sigma2*np.eye(len(self.y))) @ ktx
        return mu, np.sqrt(sigma)
    

    def addObs(self,x,y): 
        # extend the K matrix
        sigma12 = self.k(x)
        # print("k shape",sigma12)
        # print("K shape",self.K)
        kxx = np.array(self.kernel(x,x))
        if(not min(self.K.shape)):
            self.K = kxx.reshape((1,1))
        else:
            self.K = np.concatenate([
                        np.concatenate([self.K,     sigma12],axis = 1),
                        np.concatenate([sigma12.T,  kxx.reshape((1,1))],axis = 1),
                ],axis = 0)

        self.kfunlist.append(lambda x_: self.kernel(x,x_))
        self.y.append(y)




def Level_set_estimation(D, kernel, f, sigma, h, accuracy, max_iter, beta_sqrt = 1.96):
    """
        Input: sample set D(list), GP prior (µ0, k, σ0), threshold value h, accuracy parameter epsilon
        Output: predicted sets H , L
    """
    H,L = [],[]
    Cu,Cl = np.ones(len(D))*np.inf, -np.ones(len(D))*np.inf # lower and upper limit
    U = set(np.arange(0,len(D))) # the set of indexes of yet distinguished points
    # obs_x,obs_y = [],[] # observed list of x and y
    Q = GaussionProcess(kernel,sigma)
    iterations = 0
    while(len(U)):
        xind = np.argmax([min(u-h,h-l) for u,l in zip(Cu,Cl)]) # select the x to test
        y = f(D[xind])
        # y = np.exp(-y) # to revert from -log(d) in optimization problem to d in Gaussian process
        print("initState: ",D[xind], "\n f value: ", y)
        logger.add(D[xind], y)
        Q.addObs(D[xind],y)
        for xi in list(U): # loop over the unclassified points to update region
            mu, s = Q(D[xi])
            Cu[xi] = min(Cu[xi], mu+beta_sqrt*s)
            Cl[xi] = max(Cl[xi], mu-beta_sqrt*s)

            if(Cl[xi]+accuracy > h):
                U.remove(xi)
                H.append(D[xi])
                logger.amend(D[xi], "High", (Cl[xi],Cu[xi]))
                print("moved Points:",D[xi], "\nTo High Set \nits Cl and Cu:",Cl[xi],Cu[xi])
                Cl[xi] = Cu[xi] = h
            elif(Cu[xi] - accuracy <= h):
                U.remove(xi)
                L.append(D[xi])
                logger.amend(D[xi], "Low", (Cl[xi],Cu[xi]))
                print("moved Points:",D[xi], "\nTo Low Set \nits Cl and Cu:",Cl[xi],Cu[xi])
                Cl[xi] = Cu[xi] = h
        iterations += 1
        if(iterations > max_iter):
            break
    return H,L






if __name__ == "__main__":
    ## There is still a problem that the optimizer fails to find a solution that maximizes the distance
    # print("first")
    # optimize([1,1,-0.3,-0.3])
    # optimize([1,1,-0.3,-0.3])
    # optimize([0.33333333,  0.33333333, -1.66666667,  3. ]) # this yielded nan
    print(optimize([-0.78947368, -2.36842105,  1.42105263,  3.        ]))
    # print(optimize([2,2,-2,-2]))
    

    # print("second")
    # optimize([1,1,-0.1,-0.1])
    with DataLogger(name="tmp") as logger:
        kernel = lambda x,y: np.exp(-np.linalg.norm(x-y))
        D = np.concatenate([v.reshape(-1,1) for v in np.meshgrid(*[np.linspace(-3,3,20) for i in range(4)])], axis = 1)
        H,L = Level_set_estimation(D,kernel,optimize,sigma=0.01,h=0,accuracy = 0.1,max_iter=100)

        print("H\n",len(H))
        print("L\n",len(L))


    ## Test the sample function in 2D
    # D = np.concatenate([v.reshape(-1,1) for v in np.meshgrid(*[np.linspace(-3,3,100) for i in range(2)])], axis = 1)
    # kernel = lambda x,y: np.exp(-np.linalg.norm(x-y))
    # H,L = Level_set_estimation(D,kernel,lambda x: np.linalg.norm(x)-1,sigma=0.1,h=0,accuracy = 0.1,max_iter=500)
