"""
This file implements a second order SVM that classifies between safe points and unsafe points

The connection between samples and CBF
"""

import numpy as np
from sklearn.svm import SVC
from glob import glob
import pickle as pkl
import os

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

    # loaddir = "./data/tmp"
    loaddir = "./data/exp1"
    X = []
    y = []
    for f in glob(os.path.join(loaddir,"*.pkl")):
        print(f)
        data = pkl.load(open(f,"rb"))
        X += [s['testPoint'] for s in data]
        y += [ 1+ (s['fvalue']>0) for s in data]
    
    A,b,c = SVM_factors(np.array(X),y)
    
    def func(p):
        p = np.array(p)
        return (p.T @ A @ p + b.T @ p + c)


    # viz
    import matplotlib.pyplot as plt
    posX = np.array([x for x,y in zip(X,y) if y ==1])
    assert(all([func(x) < 0 for x in posX]))
    print(posX)
    negX = np.array([x for x,y in zip(X,y) if y ==2])
    assert(all([func(x) > 0 for x in negX]))

    # print(posX)
    plt.plot(posX[:,0],posX[:,1], "x", alpha = 0.4, label = "positive points")
    plt.plot(negX[:,0],negX[:,1], "x", alpha = 0.4, label = "negative points")

    N = 36
    Eclps = A[:2,:2] # the ecllips in the first two dim
    ep = []
    for theta in np.arange(0,(2+1./N)*3.14,2*3.14/N):
        tmpx = np.array([np.cos(theta),np.sin(theta)])
        # print(tmpx.T @ Eclps @ tmpx)
        a = 1 / np.sqrt(-tmpx.T @ Eclps @ tmpx)
        ep.append(tmpx * a)
    ep = np.array(ep)
    plt.plot(ep[:,0],ep[:,1],label = "barrier boundary")
    plt.legend()
    plt.show()