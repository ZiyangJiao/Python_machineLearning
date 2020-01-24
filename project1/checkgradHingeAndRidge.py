import numpy as np
def checkgradHingeAndRidge(f, X, e, x, y,lambdaa):
# % checkgradHingeAndRidge checks the derivatives in a hinge or ridge function, by comparing them to finite
# % differences approximations. The partial derivatives and the approximation
# % are printed and the norm of the difference divided by the norm of the sum is
# % returned as an indication of accuracy.
# %
# % usage: checkgradHingeAndRidge('f', X, e, x, y,lambdaa)
# %
# % where X is the argument and e is the small perturbation used for the finite
# % differences. and the x, y,lambdaa are parameters which
# % get passed to hinge or ridge function. The function hinge or ridge function should be of the type
# %
# % fX, dfX = hinge(X, x, y,lambdaa) or fX, dfX = ridge(X, x, y,lambdaa)
# %
# % where fX is the function value and dfX is a vector of partial derivatives.
# %
# % Carl Edward Rasmussen, 2001-08-01.


    y0,dy = f(X,x,y,lambdaa)   # get the partial derivatives dy
    dh = np.zeros((len(X),1))
    for j in range(len(X)):
        dx = np.zeros((len(X),1))
        dx[j] = dx[j] + e                            # perturb a single dimension
        y2,dy2 = f(X+dx,x,y,lambdaa)
        dx = -dx
        y1,dy1 = f(X+dx,x,y,lambdaa)
        dh[j] = (y2 - y1)/(2*e)

    # dh (the gradient calculated by the finite difference method) should be almost the same as dy (the gradient calculated by your function)
    print("dh:", dh)
    print("dy:", dy)

    d = np.linalg.norm(dh-dy)/np.linalg.norm(dh+dy);       # return norm of diff divided by norm of sum
    return d
