import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Bidirectional
from Evaluation import evaluation


def Model_Bi_GRU(Train_Data, Train_Target, Test_Data, Test_Target, activation):
    IMG_SIZE = 256
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE * 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE * 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE * 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE * 3))

    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(Bidirectional(
        GRU(units=50, return_sequences=True, input_shape=(Train_X.shape[1], Train_X.shape[2]), activation=activation)))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(Bidirectional(
        GRU(units=50, return_sequences=True, input_shape=(Train_X.shape[1], Train_X.shape[2]), activation='tanh')))
    regressorGRU.add(Dropout(0.2))
    # Third GRU layer
    regressorGRU.add(Bidirectional(
        GRU(units=50, return_sequences=True, input_shape=(Train_X.shape[1], Train_X.shape[2]), activation='tanh')))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(Bidirectional(GRU(units=50, activation='tanh')))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=Train_Target.shape[1]))
    regressorGRU.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Fitting to the training set
    regressorGRU.fit(Train_X, Train_Target, epochs=100, steps_per_epoch=100)
    pred = regressorGRU.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)
    return Eval, pred



