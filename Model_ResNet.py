import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout

from Evaluation import Reg_evaluation


def Model_ResNet(Train_Data, Train_target, Test_data, Test_Target, Epoch):
    IMG_SIZE = [256, 256]
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE[0] * IMG_SIZE[1], 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE[0], IMG_SIZE[1], 3))

    Test_X = np.zeros((Test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], 3))
    for i in range(Test_data.shape[0]):
        temp = np.resize(Test_data[i], (IMG_SIZE[0] * IMG_SIZE[1], 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE[0], IMG_SIZE[1], 3))

    base_model = tf.keras.applications.ResNet101(input_shape=(Train_X.shape[1], Train_X.shape[2], Train_X.shape[3]),
                                                 include_top=False,
                                                 weights="imagenet")

    for layer in base_model.layers:
        layer.trainable = False

    # Building Model
    model = Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(Train_target.shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, Train_target, epochs=Epoch, batch_size=4, steps_per_epoch=100)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = Reg_evaluation(Test_Target, pred)

    return Eval, pred