import numpy as np
from Glob_Vars import Glob_Vars
from Model_SRFCAA_LSTM import Model_SRFCAA_LSTM

def Objective_cls(Soln):
    image = Glob_Vars.Images
    target = Glob_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        learnper = round(image.shape[0] * 0.75)
        train_data = image[learnper:, :]
        train_target = target[learnper:, :]
        test_data = image[:learnper, :]
        test_target = target[:learnper, :]
        Eval = Model_SRFCAA_LSTM(train_data, train_target, test_data, test_target, sol.astype('int'))
        Fitn[i] = 1 / Eval[0]
    return Fitn
