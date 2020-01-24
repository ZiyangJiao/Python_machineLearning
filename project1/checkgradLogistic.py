import numpy as np
def checkgradLogistic(f, X, e, x, y):
# % checkgradLogistic checks the derivatives in the Logistic function, by comparing them to finite
# % differences approximations. The partial derivatives and the approximation
# % are printed and the norm of the difference divided by the norm of the sum is
# % returned as an indication of accuracy.
# %
# % usage: checkgrad('f', X, e, x, y)
# %
# % where X is the argument and e is the small perturbation used for the finite
# % differences. and the x, y are  additional parameters which
# % get passed to Logistic. The Logistic function should be of the type 
# %
# % fX, dfX = logistic(X, x, y)
# %
# % where fX is the function value and dfX is a vector of partial derivatives.
# %
# % Carl Edward Rasmussen, 2001-08-01.


    y0,dy = f(X,x,y)   # get the partial derivatives dy
    dh = np.zeros((len(X),1)) 
    for j in range(len(X)):
        dx = np.zeros((len(X),1)) 
        dx[j] = dx[j] + e                            # perturb a single dimension
        y2,dy2 = f(X+dx,x,y)
        dx = -dx
        y1,dy1 = f(X+dx,x,y)
        dh[j] = (y2 - y1)/(2*e)
    
    # dh (the gradient calculated by the finite difference method) should be almost the same as dy (the gradient calculated by your function)
    # print("dh:", dh)
    # print("dy:", dy)
    
    d = np.linalg.norm(dh-dy)/np.linalg.norm(dh+dy);       # return norm of diff divided by norm of sum
    return d
