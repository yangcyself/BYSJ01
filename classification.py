"""
This file implements a second order SVM that classifies between safe points and unsafe points

The connection between samples and CBF
"""

import numpy as np
from sklearn.svm import SVC
# X = np.array([[-1, -1], [0, 0], [1, 1]])
# y = np.array([1, 2, 1])
X = np.random.random((4,4))
y = np.array([1,1,2,2])
clf = SVC(kernel = "poly", degree=2, gamma=99999 )
clf.fit(X, y)

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

A,b,c = recoverSecondOrderFunc(lambda x : clf.decision_function(x.reshape((1,-1))),dim = 4)
print(A,b,c)

def func(p):
    p = np.array(p)
    return (p.T @ A @ p + b.T @ p + c)

def test(p):
    p = np.array(p)
    print(clf.decision_function(p.reshape(1,-1)), func(p))

for i in range(10):
    test(np.random.random(4))