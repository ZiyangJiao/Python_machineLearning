from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    res = w.T @ xTr
    tmp = np.zeros((len(yTr), 1))
    for i in range(len(yTr)):
        if 1 - yTr[0][i] * res[0][i] > 0:
            tmp[i] = 1 - yTr[0][i] * res[0][i]
        else:
            tmp[i] = 0
    loss = np.sum(tmp) + lambdaa * w.T @ w
    for i in range(len(yTr)):
        if yTr[0][i] * res[0][i] > 1:
            yTr[0][i] = 0
    gradient = 2 * lambdaa * w - xTr @ yTr.T

    return loss,gradient
