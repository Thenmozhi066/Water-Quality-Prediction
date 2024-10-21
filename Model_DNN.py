import numpy as np
import random as random

from Neural_Network import NeuralNetwork

from Evaluation_All import evaluation


def Model_DNN(data, labels, test_data, test_target, sol=None):
    if sol is None:
        sol = [10, 0.5]

    if len(data.shape) == 3:
        data = np.resize(data, (data.shape[0], data.shape[1] * data.shape[2]))
        test_data = np.resize(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))


    simple_network = NeuralNetwork(no_of_in_nodes=data.shape[1],
                                   no_of_out_nodes=10,
                                   no_of_hidden_nodes=data.shape[1],  # 5,
                                   learning_rate=0.5,
                                   sol=int(sol[0]),
                                   bias=None)

    for _ in range(20):
        for i in range(len(data)):
            simple_network.train(data[i, :], labels[i])

    pred = simple_network.run(test_data)
    predict = np.zeros((pred.shape[1])).astype('int')
    for i in range(pred.shape[1]):
        if pred[0, i] > 0.5:
            predict[i] = 1
        else:
            predict[i] = 0
    Pred = np.round(predict)
    Eval = evaluation(Pred.reshape(-1, 1), test_target)
    return Eval


