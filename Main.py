import numpy as np
import pandas as pd
from numpy import matlib
import random as rn
from sklearn.utils import shuffle
from DOA import DOA
from Glob_Vars import Glob_Vars
from HCO import HCO
from Model_DBN import Model_DBN
from Model_DNN import Model_DNN
from Model_GRU import Model_GRU
from Model_SRFCAA_LSTM import Model_SRFCAA_LSTM
from OOA import OOA
from Objective_Function import Objective_cls
from POA import POA
from PROPOSED import PROPOSED
from Plot_Results import plot_results_conv, Plot_ROC, Plot_Met, Plot_Alg, Plot_table

no_of_dataset = 2

# Read Dataset 1
an = 0
if an == 1:
    dir = './Dataset/Dataset 1/dataset.csv'
    read = pd.read_csv(dir)
    read_data = read.values
    list_ = [7, 17, 20]
    count = 5000
    for a in range(len(list_)):
        replace_column1 = read_data[:, list_[a]]
        Targ = replace_column1.astype('str')
        uni = np.unique(Targ)
        for i in range(len(uni)):
            ind = np.where((Targ == uni[i]))
            read_data[ind[0], list_[a]] = i
    Tar = read_data[:, -1].astype('int')
    read_datas = pd.DataFrame(read_data[:, :-1])
    datas = read_datas.values
    for b in range(len(datas)):
        ind = np.where(('nan' == datas[b, :].astype('str')))
        datas[b, ind[0]] = 0
    uniq, counts = np.unique(Tar, return_counts=True)
    index_0 = np.where(Tar == uniq[0])[0]
    index_1 = np.where(Tar == uniq[1])[0]
    Target = np.append(Tar[index_0[:count]], Tar[index_1[:count]]).reshape(-1, 1)
    Data = np.append(datas[index_0[:count]], datas[index_1[:count]], axis=0)
    Data, Target = shuffle(Data, Target)
    np.save('Data_1.npy', Data)
    np.save('Target_1.npy', Target)

# Read Dataset 2
an = 0
if an == 1:
    dir = './Dataset/Dataset 2/water_potability.csv'
    read = pd.read_csv(dir)
    read_data = read.values
    Target = read_data[:, -1].astype('int')
    read_datas = pd.DataFrame(read_data[:, :-1])
    datas = read_datas.values
    for b in range(len(datas)):
        ind = np.where(('nan' == datas[b, :].astype('str')))
        datas[b, ind[0]] = 0
    np.save('Data_2.npy', datas.astype('float'))
    np.save('Target_2.npy', Target.reshape(-1, 1))

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Optimization for Classification $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
an = 0
if an == 1:
    sol = []
    fitness = []
    for i in range(no_of_dataset):
        Feat = np.load('Data_' + str(i + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(i + 1) + '.npy', allow_pickle=True)
        Glob_Vars.Feat = Feat
        Glob_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat(([5, 5, 300]), Npop, 1)
        xmax = matlib.repmat(([255, 50, 1000]), Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
        fname = Objective_cls
        max_iter = 50

        print('POA....')
        [bestfit1, fitness1, bestsol1, Time1] = POA(initsol, fname, xmin, xmax, max_iter)

        print('HCO....')
        [bestfit2, fitness2, bestsol2, Time2] = HCO(initsol, fname, xmin, xmax, max_iter)

        print('DOA....')
        [bestfit3, fitness3, bestsol3, Time3] = DOA(initsol, fname, xmin, xmax, max_iter)

        print('OOA....')
        [bestfit4, fitness4, bestsol4, Time4] = OOA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        sol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('Bestsol.npy', sol)
    np.save('Fitness.npy', fitness)

## Classification ##
an = 0
if an == 1:
    Eval = []
    for k in range(no_of_dataset):
        Feat = np.load('Data_' + str(k + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(k + 1) + '.npy', allow_pickle=True)
        sol = np.load('Bestsol.npy', allow_pickle=True)[k][4, :]
        ACT = ['Linear', 'ReLU', 'Leaky ReLU', 'TanH', 'Sigmoid', 'Softmax']
        vl = [0, 1, 2, 3, 4]
        for m in range(len(ACT)):
            per = round(Feat.shape[0] * 0.75)
            EVAL = np.zeros((10, 25))
            for i in range(5):  # for all algorithms
                train_data = Feat[:per, :]
                train_target = Target[:per, :]
                test_data = Feat[per:, :]
                test_target = Target[per:, :]
                # EVAL[i, :] = Model_SRFCAA_LSTM(train_data, train_target, test_data, test_target, sol[i].astype('int'))
            train_data = Feat[:per, :]
            train_target = Target[:per, :]
            test_data = Feat[per:, :]
            test_target = Target[per:, :]
            # EVAL[5, :] = Model_DNN(train_data, train_target, test_data, test_target)
            EVAL[6, :] = Model_DBN(train_data, train_target, test_data, test_target)
            EVAL[7, :] = Model_GRU(train_data, train_target, test_data, test_target)
            EVAL[8, :] = Model_SRFCAA_LSTM(train_data, train_target, test_data, test_target)
            EVAL[9, :] = EVAL[4, :]
            Eval.append(EVAL)
    np.save('Eval_pred.npy', Eval)

plot_results_conv()
Plot_ROC()
Plot_Met()
Plot_Alg()
Plot_table()
