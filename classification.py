"""
This file implements a second order SVM that classifies between safe points and unsafe points

The connection between samples and CBF
"""

import numpy as np
from sklearn.svm import SVC
from glob import glob
import pickle as pkl
from scipy.optimize import minimize
import os
from problemSetting import *
import json
from util.visulization import drawEclips

def recoverSecondOrderFunc(f,dim = 4):
    # retrive x^T A x + b^T x + c = decision_func
    c = f(np.zeros(dim))
    b = np.zeros(dim)
    A = np.zeros((dim,dim))
    for i in range(dim):
        t1,t2 = np.zeros(dim),np.zeros(dim)
        t1[i] = 1
        t2[i] = -1
        b[i] = (f(t1)-f(t2))/2
        A[i,i] = (f(t1)+f(t2))/2 - c
    for i in range(dim):
        for j in range(i):
            t = np.zeros(dim)
            t[i] = t[j] = 1
            A[i,j] = A[j,i] = (f(t) - t.T @ A @ t - b @ t - c)/2
    return A,b,c

def SVM_factors(X,y,dim = 4):
    clf = SVC(kernel = "poly", degree=2, gamma=99999 )
    clf.fit(X, y)
    A,b,c = recoverSecondOrderFunc(lambda x : clf.decision_function(x.reshape((1,-1))),dim = dim)
    print("Function parameter \n",A,b,c)
    return A,b,c



# Add more constraints to the fitting problem

class kernel:
    """
        a map to some higher dim space. E.g (x,y) -> (x^2,y^2,x,y)
    """
    @staticmethod
    def augment(x):
        """
            input is either 1dim dataponit, or a matrix where each row is a datapoint
            return the augmented matrix, where each row is the feature of a datapoint
        """
        x = np.array(x)
        if(x.ndim==1):
            x  = x.reshape((1,-1))
        x = x.T
        return np.concatenate([[a*b for i,a in enumerate(x) for b in x[i:] ],x,np.ones((1,x.shape[1]))],axis = 0).T

    @staticmethod
    def jac(x):
        """
            input is 1dim dataponit
            return the jacobian of this datapoint
        """
        x = np.array(x)
        return np.concatenate([ [ [(a if k==j+i else 0) + (b if k==i else 0)  for k in range(len(x))] 
                                for i,a in enumerate(x) for j,b in enumerate(x[i:]) ] ,np.eye(len(x)),np.zeros((1,len(x)))],axis=0)
    
    @staticmethod
    def GetParam(w,dim=4):
        """
            input the trained w
            Return the A, b, c of x^TAx + b^T x + c
        """
        print("w:",w)
        A = np.array([[w[min(i,j)*dim - int(min(i,j)*(min(i,j)-1)/2) - min(i,j) + max(i,j)]/(1 if(i==j)else 2)  for j in range(dim)] for i in range(dim)])
        b = w[-dim-1:-1]
        c = w[-1]
        return A,b,c



def fitCBF(X, y, X_c,dim = 4, x0 = None):
    """
        X: tested datapoints
        y: 1 or -1; 1 means in CBF's upper level set
        X_c: The datapoints that needs to ensure exits u s.t. dB(x,u) > 0

        cast it into an optimization problem, where
        min_{w,b,u}  ||w||+||u||
        s.t.    y_i(x_i^T w + b) > 1 //SVM condition
                y_i(dB(x_i,u)+ mc B(x_i)) > 0 for y_i > 0  //CBF defination
                u_min < u < u_max
    """
    
    print("START")
    def obj(w):
        return sqeuclidean(w[:-1])
    def grad(w):
        ans = 2*w
        ans[-1] = 0
        return ans.reshape((1,-1))

    X_aug = kernel.augment(X)
    y = np.array(y).reshape((-1))

    def SVMcons(w):
        return y * (X_aug @ w) - 1
    def SVMjac(w):
        # print(w)
        return y.reshape((-1,1))*X_aug

    # X_c_aug = kernel.augment(X_c) if len(X_c) else []

    def completeCons(w):
        # the points at the boundary of the CBF needs to have a solution that dB > 0
        return np.array([1]+[w.T @ kernel.jac(x) @ Dyn_A @ x  +  
                    MAX_INPUT * np.linalg.norm(Dyn_B.T @ kernel.jac(x).T @ w, ord = 1) for x in X_c])

    options = {"maxiter" : 500, "disp"    : True}
    lenx0 = int((dim+1)*dim/2 + dim + 1)
    x0 = np.random.random(lenx0) if x0 is None else x0
    
    constraints = [{'type':'ineq','fun':SVMcons, "jac":SVMjac},
                   {'type':'ineq','fun':completeCons}]
    
    bounds = np.ones((lenx0,2)) * np.array([[-1,1]]) * 100
    bounds[-5:-1,:] *=0

    res = minimize(obj, x0, options = options,jac=grad, bounds=bounds,
                constraints=constraints, method =  'SLSQP') # 'trust-constr' , "SLSQP"

    # print("SVM Constraint:\n", SVMcons(res.x[:len(x0)]))
    return (*kernel.GetParam(res.x), res.x)


def fitCompleteCBF(X,y,dim = 4):
    """
        call the `fitCBF` iteratively, each step augment the dataset with some data points 
            that makes the optimization of u hard
    """
    x0 = None
    X_c = []
    for i in range(10):
        A,b,c, x0 = fitCBF(X,y,X_c,dim,x0)
    
        def obj(x):
            return x.T @ (Dyn_A .T @ A + A @ Dyn_A) @ x + 2 * MAX_INPUT * np.linalg.norm(Dyn_B.T @ A @ x,ord = 1)

        def cons(x):
            return x.T @ A @ x - c

        constraints = {'type':'eq','fun': cons, "jac":jacobian(cons)}
        
        bounds = np.ones((4,2)) * np.array([[-1,1]]) * 3
        
        X_c = [ minimize(obj, np.random.random(4) * 3, bounds = bounds, #jac=jac, 
                        constraints=constraints, method =  'SLSQP').x for i in range(1) ]
        print([obj(x) for x in X_c])
        # print(X_c)
    return A,b,c

    # def CBFCons(state,w,u):
    #     """
    #         the CBF condition, at the state
    #         dB(x,u) + mc B(x) > 0
    #     """
    #     mc = 1
    #     # print(w.shape,kernel.augment(state).shape)
    #     B = kernel.augment(state) @ w
    #     # print(kernel.jac(state).shape)
    #     # print((w.T @ kernel.jac(state)).shape)
    #     # print(Dyn_A @ state)
    #     # print(Dyn_B @ u)
    #     # print( (Dyn_A @ state.reshape((-1,1)) + Dyn_B @ u.reshape(-1,1)).shape)
    #     dB = w.T @ kernel.jac(state) @ (Dyn_A @ state + Dyn_B @ u)
    #     return dB + mc * B

    # def CBFjac(state,w,u):
    #     mc = 1
    #     return (kernel.augment(state) + mc * kernel.jac(state) @ (Dyn_A @ state + Dyn_B @ u), # jac of w
    #             (mc *  (w.T @ kernel.jac(state) @ Dyn_B ).T).reshape(1,-1) ) # jac of u



def dumpJson(A,b,c,fileName = "data/tmp/Abc.json"):
    json.dump({"A":A.tolist(),"b":b.tolist(),"c":c},open(fileName,"w"))


if __name__ == "__main__":
    #### TEST recoverSecondOrderFunc
    # X = np.array([[-1, -1], [0, 0], [1, 1]])
    # y = np.array([1, 2, 1])
    # X = np.random.random((4,4))
    # y = np.array([1,1,2,2])
    

    # def test(p):
    #     p = np.array(p)
    #     print(clf.decision_function(p.reshape(1,-1)), func(p))

    # for i in range(10):
    #     test(np.random.random(4))


    # ### Begin Comment
    # loaddir = "./data/tmp"
    loaddir = "./data/exp1"
    X = []
    y = []
    for f in glob(os.path.join(loaddir,"*.pkl")):
        print(f)
        data = pkl.load(open(f,"rb"))
        X += [s['testPoint'] for s in data]
        y += [ 1 if (s['fvalue']<0) else -1 for s in data]
        # y += [ 1 if (s['fvalue']>0) else -1 for s in data]
    augmentData = [[np.cos(th),np.sin(th),3*np.cos(th),np.sin(th)] for th in np.arange(0,2*3.14,20)]
    X += augmentData
    y += [-1] * len(augmentData)
    # A,b,c = SVM_factors(np.array(X),y)
    A,b,c = fitCompleteCBF(X,y)
    # A,b,c = kernel.GetParam(w,dim=4)

    c = float(c)
    print(A,b,c)
    dumpJson(A,b,c,"data/exp1/svm_complete.json")
    # dumpJson(A,b,c,"data/exp1/svm_def_aug.json")
    # dumpJson(A,b,c,"data/tmp/svm_def.json")
    
    def func(p):
        p = np.array(p)
        return (p.T @ A @ p + b.T @ p + c)

    # # viz
    import matplotlib.pyplot as plt
    posX = np.array([x for x,y in zip(X,y) if y ==1])
    assert(all([func(x) > -1e-9 for x in posX]))
    # print(posX)
    negX = np.array([x for x,y in zip(X,y) if y ==-1])
    assert(all([func(x) < 1e-9 for x in negX]))

    plt.plot(posX[:,0],posX[:,1], "x", alpha = 0.4, label = "positive points")
    plt.plot(negX[:,0],negX[:,1], "x", alpha = 0.4, label = "negative points")

    drawEclips(A,b,c)
    plt.legend()
    plt.show()


    # print(kernel.augment([[1,2,3],
    #                       [2,4,1]]))

    # print(kernel.jac([1,2,3]))



