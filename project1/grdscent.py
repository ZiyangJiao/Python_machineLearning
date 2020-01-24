import sys
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    w = w0
    oldloss = sys.maxsize
    for i in range(maxiter):
        [newloss, gradient] = func(w)
        if stepsize < eps:
            print("Error: Stepsize is too small!")
            break
        if np.linalg.norm(gradient) < tolerance:
            break
        if newloss >= oldloss:
            stepsize *= 0.5
        else:
            stepsize *= 1.01
            
        w = w - stepsize * gradient
        oldloss = newloss
        # print(newloss)
    return w
