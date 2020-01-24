
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);
    
    # YOUR CODE HERE
    loss = (w.T @ xTr - yTr) @ (w.T @ xTr - yTr).T + lambdaa * w.T @ w
    gradient = 2 * (xTr @ xTr.T @ w - xTr @ yTr.T + lambdaa * w)
    
    return loss, gradient
