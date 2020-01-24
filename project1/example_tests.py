import numpy as np
import random
import math
from logistic import logistic
from ridge import ridge
from checkgradLogistic import checkgradLogistic
from checkgradHingeAndRidge import checkgradHingeAndRidge

def example_tests():
# % def example_tests():
# %
# % Tests for the SRM project. Some few example tests are implemented.
# % Some are only dewscribed in the comments. You will have to implement
# % those yourself.
# %
# % Output:
# % r:    number of tests that broke
# % ok:   number of passed tests
# % s:    statement describing the failed test (s={} if all succeed)


    random.seed(31415926535)
    # % initial outputs
    r=0
    ok=0
    s=[]  #used to be matlab cell array

    # data set
    N=50
    D=5

    x=np.concatenate((np.random.randn(D,N),np.random.randn(D,N)+2),axis=1)
    y=np.concatenate((np.ones((1,N)),-np.ones((1,N))),axis=1)

    print ('Starting Test 1\n')
    #Test 1: testing gradient of logistic
    d=checkgradLogistic(logistic,np.random.rand(D,1),1e-05,x,y)
    failtest = d>1e-10

    if failtest:
        r=r+1
        s.append('Test 1: Logistic function does not pass checkgrad.')
    else:
        ok=ok+1;

    print('Completed Test 1\n')

#     %% Test 2: logistic sanity check #1
#     % we will test logistic with an all zeros weight vector, a random datapoint
#     % and a random label in {-1,1}. The expected outcome is (very close to) log(2).

    print('Starting Test 3\n')
    #Test 3: logistic sanity check #2
    w=np.random.rand(5,1)
    logistic_loss = logistic(w,x[:,1].reshape((5,1)),np.ones((1,1)))[0]
    failtest = w.T.dot(x[:,1])+math.log(math.exp(logistic_loss)-1) > 2.2204e-16
    if failtest:
        r=r+1
        s.append('Test 3: Logistic function does not pass sanity check #2.')
    else:
        ok=ok+1
    print('Completed Test 3\n')


    print('Starting Test 4\n')
    #Test 4: testing gradient of ridge
    d = checkgradHingeAndRidge(ridge,np.random.rand(D, 1), 1e-05, x,y,10)
    failtest = d > 1e-10

    if failtest:
        r = r+1
        s.append('Test 4: Ridge function does not pass checkgrad.')
    else:
        ok=ok+1
    print('Completed Test 4\n')

#     %% Test 5: testing gradient of hinge
#     % we will test hinge using checkgrad on randomly generated x and y data
#     % initializing w with 1e-05 and lambda with 1e-05. The gradient is supposed
#     % to be smaller than 5e-07.
#
#
#     %% Test 6: checking gradient descent
#     % we will check grdescent using the squared loss, randomly generated input
#     % weights and stepsize=1e-05, maxiter=1000,and tolerance=1e-09. The
#     % norm of the gradient at the optimal solution should be zero (< 1e-05).
#
#
#     %% Tests 7-12: solutions of hinge, ridge, and logistic
#     % we will compare the solutions (loss value and gradient) of hinge,
#     % ridge, and logistic to our implementation using x and y.
#     % Note that you cannot implement those tests.

    return r,ok,s

def squaredloss(w,x,y):
    [d,n]=np.shape(x)
    diff=(w.T.dot(x)-y)
    gradient=2*x.dot(diff.T)/n
    loss = np.mean(diff**2)
    return loss,gradient

if __name__ == '__main__':
    failed,ok,msgs = example_tests()
    print("Number of failed example tests: "+str(failed))
    print("Number of passed example tests: "+str(ok))
    if len(msgs):
        failMsg = 'Unfortunately, you failed %d test(s) on this evaluation: \n\n' % len(msgs)
        for j in range(0,len(msgs)):
            print(msgs[j])
    print("\nNote: we only implemented 3 out of 12 tests for you. Check the inline documentation for what the other tests do and implement them yourself!")
