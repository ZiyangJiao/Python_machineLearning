

def linearmodel(w,xTe):
# INPUT:
# w weight vector (default w=0)
# xTe dxn matrix (each column is an input vector)
#
# OUTPUTS:
#
# preds predictions

    # YOUR CODE HERE
    preds = w.T @ xTe
    return preds
