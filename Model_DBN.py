# from keras.utils import to_categorical
import numpy as np
from keras import backend as K
from Evaluation_pred import error_evaluation
from dbn.tensorflow import SupervisedDBNClassification


def Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target, soln=None):
    sol = [5, 5]
    classifier = SupervisedDBNClassification(hidden_layers_structure=[sol[0], sol[1]],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=1,
                                             n_iter_backprop=2,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    pred = np.zeros((Test_Target.shape[0], Test_Target.shape[1]))
    for i in range(Test_Target.shape[1]):
        print(i)
        classifier.fit(Train_Data, Train_Target[:, 0])
        pred[:, i] = classifier.predict(Test_Data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = error_evaluation(pred, Test_Target)
    return Eval


def Model_DBN_feat(Data, Target, soln=None):
    sol = [soln, soln]
    classifier = SupervisedDBNClassification(hidden_layers_structure=[sol[0], sol[1]],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=1,
                                             n_iter_backprop=2,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    X = np.reshape(Data, (Data.shape[0], 1, Data.shape[1]))
    testX = np.reshape(Data, (Data.shape[0], 1, Data.shape[1]))
    benchmark_layers = classifier
    benchmark_input = X.shape
    inp = benchmark_layers.input  # input placeholder
    outputs = [layer.output for layer in benchmark_layers.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 1
    Feature = []
    for i in range(X.shape[0]):
        hgn = X[i, :, :][np.newaxis, ...]
        test = hgn.reshape(-1, 1, 1)
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feature.append(layer_out)
    DBN_Feat = np.asarray(Feature)

    return DBN_Feat
