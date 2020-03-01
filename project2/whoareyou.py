import numpy as np
from classifyLinear import classifyLinear
from name2features import name2features

def whoareyou(w,b=0):
# =============================================================================
# function whoareyou(w,b);
# 
# A little interactive demo of your name classifier
# =============================================================================

    while True:
        name=input('Who are you>');
        if (name=='byebye') == 1:
            break
        x = name2features(name)
        x = x.reshape((x.shape[0],1))
        pred=classifyLinear(x,w,b)
        if pred > 0:
            out=name + ', I am sure you are a nice girl.'
        else:
            out=name + ', I am sure you are a nice boy.'
        print(out)
