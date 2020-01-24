import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE
    e = np.exp(-1*yTr*(w.T@xTr))
    loss = np.sum(np.log(1+e))
    gradient = xTr @ ((-1*yTr*e)/(1+e)).T
    return loss,gradient
