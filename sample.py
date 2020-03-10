"""
    The pipeline for sample is:
        randomly select points(states)
        run optimization for each point
"""
from problemSetting import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import copy


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

    def objective(dyn_u):
        return np.linalg.norm(dyn_u)
        # print(-np.min([ np.min(constraintOftTraj(c)(dyn_u)) for c in collisionList]))
        # return -np.min([ np.min(constraintOftTraj(c)(dyn_u)) for c in collisionList])


    constraints = [{'type':'ineq','fun': constraintOftTraj(c)} for c in collisionList]

    # x0 = np.zeros(2*horizon)
    x0 = np.ones(2*horizon)
    bounds = np.ones((2*horizon,2)) * np.array([[-1,1]])
    options = {"maxiter" : 500, "disp"    : 2}
    res = minimize(objective, x0, bounds=bounds,options = options,
                constraints=constraints)

    # constraintViolation = np.linalg.norm(np.clip([c['fun'](res.x) for c in constraints],None,0)) 
    print("solution:",res.x[:10])
    constraintViolation = -np.min([ np.min(constraintOftTraj(c)(res.x)) for c in collisionList])
    print("constraint violation:", constraintViolation)
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
        self.k = lambda x:np.array([f(x) for f in self.kfunlist]) # the function that returns the vector of a point ot observed points
        self.y = []


    def __call__(self,x):
        # return the posterior mu and sigma inferred from current gaussian process
        ktx = np.linalg.inv(self.K+self.sigma2*np.eye(len(self.y)))
        mu = ktx.T @ np.linalg.inv(self.K+self.sigma2*np.eye(len(self.y))) @ np.array(self.y)

        sigma = self.kernel(x,x) - ktx.T @ np.linalg.inv(self.K+self.sigma2*np.eye(len(self.y))) @ ktx
        return mu, np.sqrt(sigma)
    

    def addObs(self,x,y):
        self.kfunlist.append(lambda x_: self.kernel(x,x_))
        
        # extend the K matrix
        sigma12 = self.k(x)
        kxx = np.array(self.kernel(x,x))
        self.K = np.concatenate([
                    np.concatenate([self.K,     sigma12],axis = 1),
                    np.concatenate([sigma12.T,  kxx.reshape((1,1))],axis = 1),
            ],axis = 0)

        self.y.append(y)

kernel = lambda x,y: np.exp(-np.linalg.norm(x-y))


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
    while(len(U)):
        xind = np.argmax(Cu - Cl) # select the x to test
        y = f(D[xind])
        Q.addObs(D[xind],y)
        for xi in U: # loop over the unclassified points to update region
            mu, s = Q(D(xi))
            Cu[xi] = min(Cu[xi], mu+beta_sqrt*s)
            Cl[xi] = max(Cl[xi], mu-beta_sqrt*s)

            if(Cl[xi]+accuracy > h):
                U.remove(xi)
                H.append(D[xi])
                Cl[xi] = Cu[xi] = h
            elif(Cu[xi] - accuracy <= h):
                U.remove(xi)
                L.append(D[xi])
                Cl[xi] = Cu[xi] = h
                
    return H,L






if __name__ == "__main__":
    print("first")
    optimize([1,1,-0.3,-0.3])

    print("second")
    optimize([1,1,-0.1,-0.1])
    # D = np.concatenate([v.reshape(-1,1) for v in np.meshgrid(*[np.linspace(-3,3,10) for i in range(4)])], axis = 1)
    # # print(D)
    # # print(D.shape)
    # H,L = Level_set_estimation(D,kernel,optimize,sigma=0.1,h=0,accuracy = 0.1,max_iter=10)
    # print("H\n",H)
    # print("L\n",L)
