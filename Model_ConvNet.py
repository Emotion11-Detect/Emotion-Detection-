from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense
import numpy as np

from Evaluation import evaluation


def Model_ConvNet(Train_Data, Train_Target, Test_Data, Test_Target, activation):
    IMG_SIZE = 32
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE * 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE * 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE * 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE * 3))

    model = Sequential()
    model.add(Conv1D(32, 3, padding="same", activation=activation, input_shape=(IMG_SIZE, IMG_SIZE * 3)))
    model.add(MaxPooling1D(1, strides=2))
    model.add(Conv1D(32, 3, padding="same", activation=activation))
    model.add(MaxPooling1D(1, strides=2))
    model.add(Conv1D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(1, strides=2))
    model.add(Conv1D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(1, strides=2))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(Train_Target.shape[1], activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, Train_Target, epochs=1, batch_size=32)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred

