"""
    The pipeline for sample is:
        randomly select points(states)
        run optimization for each point
"""
from problemSetting import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import copy
from  util.visulization import plotAction

def optimize(initState, horizon = 50):
    """
        Get a initstate and return the optimal constraint violation
            This serves as the sample function(f) in paper
    """
    def constraintOftTraj(c):
        def returnfunc(dyn_u):
            result = np.zeros(horizon)
            x = initState
            for i in range(horizon):
                result[i] = c(x)
                x = sys_A @ x + sys_B @ dyn_u[2*i:2*i+2]
            return result
        return returnfunc
    

    def jacOfTraj(c):
        def returnfunc(dyn_u):
            result = np.zeros((horizon,len(dyn_u)))
            x = initState
            stateJac = np.zeros((4,len(dyn_u)))
            for i in range(horizon):
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
        return -np.min([ np.min(constraintOftTraj(c)(dyn_u)) for c in collisionList])


    def obj_grad(dyn_u):
        i = np.argmin([ np.min(constraintOftTraj(c)(dyn_u)) for c in collisionList])
        j = np.argmin(constraintOftTraj(collisionList[i])(dyn_u))
        # print(i,j)
        # print(-jacOfTraj(collisionList[i])(dyn_u)[j,:])
        return -jacOfTraj(collisionList[i])(dyn_u)[j,:]
        # return 2 * dyn_u

    constraints = [{'type':'ineq','fun': constraintOftTraj(c), "jac":jacOfTraj(c) } for c in collisionList]

    x0 = np.zeros(2*horizon)
    # x0 = np.ones(2*horizon)
    bounds = np.ones((2*horizon,2)) * np.array([[-1,1]]) * 10
    options = {"maxiter" : 500, "disp"    : False}
    res = minimize(objective, x0, bounds=bounds,options = options,jac=obj_grad,#)
                constraints=constraints)

    # constraintViolation = np.linalg.norm(np.clip([c['fun'](res.x) for c in constraints],None,0)) 
    print('\n initState:',initState)
    print("solution:",res.x)
    constraintViolation = -np.min([ np.min(constraintOftTraj(c)(res.x)) for c in collisionList])
    print("constraint violation:", constraintViolation)
    plotAction(initState,res.x)
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
        xind = np.argmax(Cu - Cl) # select the x to test
        y = f(D[xind])
        Q.addObs(D[xind],y)
        for xi in list(U): # loop over the unclassified points to update region
            mu, s = Q(D[xi])
            Cu[xi] = min(Cu[xi], mu+beta_sqrt*s)
            Cl[xi] = max(Cl[xi], mu-beta_sqrt*s)

            if(Cl[xi]+accuracy > h):
                U.remove(xi)
                H.append(D[xi])
                print("moved Points:",D[xi], "\n its Cl and Cu:",Cl[xi],Cu[xi])
                Cl[xi] = Cu[xi] = h
            elif(Cu[xi] - accuracy <= h):
                U.remove(xi)
                L.append(D[xi])
                print("moved Points:",D[xi], "\n its Cl and Cu:",Cl[xi],Cu[xi])
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
    optimize([3,3,-3,-3])
    

    # print("second")
    # optimize([1,1,-0.1,-0.1])
    # kernel = lambda x,y: np.exp(-np.linalg.norm(x-y))
    # D = np.concatenate([v.reshape(-1,1) for v in np.meshgrid(*[np.linspace(-3,3,10) for i in range(4)])], axis = 1)
    # # print(D)
    # # # print(D.shape)
    # H,L = Level_set_estimation(D,kernel,optimize,sigma=0.1,h=0,accuracy = 0.1,max_iter=10)

    # print("H\n",len(H))
    # print("L\n",len(L))
