import numpy as np
from keras.applications import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model
from Evaluation import evaluation


def Model_ResNet50(Train_Data, Train_target, Test_data, Test_Target, Activation):
    IMG_SIZE = [256, 256]
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE[0] * IMG_SIZE[1], 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE[0], IMG_SIZE[1], 3))

    Test_X = np.zeros((Test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], 3))
    for i in range(Test_data.shape[0]):
        temp = np.resize(Test_data[i], (IMG_SIZE[0] * IMG_SIZE[1], 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE[0], IMG_SIZE[1], 3))

    resnet = ResNet50(
        input_shape=IMG_SIZE + [3],
        weights='imagenet',
        include_top=False
    )
    x = Flatten()(resnet.output)
    prediction = Dense(Train_target.shape[1], activation=Activation)(x)
    model = Model(inputs=resnet.input, outputs=prediction)
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(Train_X, Train_target, epochs=100, batch_size=4, steps_per_epoch=100)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)

    return Eval, pred

