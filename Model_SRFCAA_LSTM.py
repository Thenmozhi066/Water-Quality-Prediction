import numpy as np
from tf_keras import Sequential, Input
from tf_keras.src.layers import Dense, LSTM

from Evaluation_All import evaluation


def Spatial_Temporal(Train_X, sol):
    # Synthetic spatial data (e.g., 10x10 grids)
    spatial_data = np.random.rand(Train_X.shape[0], Train_X.shape[1], Train_X.shape[2], 1)

    # Synthetic temporal data (e.g., 5 time steps, 2 features)
    temporal_data =np.random.rand(Train_X.shape[0], 5, 300) # np.random.rand(Train_X.shape[0], sol[0], sol[1])

    # Spatial data processing
    spatial_input = Input(shape=(spatial_data.shape[0], temporal_data.shape[1], 1), name='spatial_input')

    return spatial_input


def Model_SRFCAA_LSTM(train_data, train_target, test_data, test_target, sol = None):
    print('SRFCAA_LSTM')
    if sol is None:
        sol = [5, 5, 300]
    out = LSTM_train(train_data, train_target, test_data, test_target, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred.reshape(-1, 1), test_target)
    return Eval


def LSTM_train(trainX, trainY, testX, testy, sol):
    IMG_SIZE = [1, 100]
    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])
    Train_X = np.array(Train_X)

    Test_Temp = np.zeros((testy.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(testy.shape[0]):
        Test_Temp[i, :] = np.resize(testy[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])
    Test_X = np.array(Test_X)
    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax', 'leaky relu']
    Spatial_Temporal(Train_X, sol)
    model = Sequential()
    classes = trainY.shape[-1]
    model.add(LSTM(10, input_shape=(Train_X.shape[1], Train_X.shape[-1])))  # hidden neuron count(5 - 255)
    model.add(Dense(50, activation="relu"))
    model.add(Dense(classes, activation="relu"))  # activation="relu"
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(Train_X, trainY, epochs=2, steps_per_epoch=2, batch_size=2, verbose=1.0)
    pred = model.predict(Test_X)
    return pred
