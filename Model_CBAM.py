import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras import layers

from Evaluation import evaluation


def cbam_block(cbam_feature, ratio=8):  # Convolutional Block Attention Module CBAM
    """
    Convolutional Block Attention Module (CBAM) block implementation.
    Enhances the feature representation by focusing on important features in channel and spatial dimensions.
    Reference: https://arxiv.org/abs/1807.06521.
    """
    cbam_feature = channel_attention(cbam_feature, ratio)  # Apply channel attention
    cbam_feature = spatial_attention(cbam_feature)  # Apply spatial attention
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    """
    Applies channel attention mechanism to the input feature map.
    Focuses on 'what' is meaningful given an input.
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # Determine the channel axis
    channel = input_feature.shape[channel_axis]  # Get the number of channels

    # Shared MLP layers
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    # Average pooling path
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    # Max pooling path
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    # Combine features from average and max pooling paths
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    # Adjust feature map for 'channels_first' data format
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    """
    Applies spatial attention mechanism to the input feature map.
    Focuses on 'where' is an informative part in the feature map.
    """
    kernel_size = 7  # Kernel size for the convolution operation in spatial attention

    # Adjusting input feature based on channel data format
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    # Apply average and max pooling
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    # Convolution layer for spatial attention
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)

    # Adjust feature map for 'channels_first' data format
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def Model_CBAM(Train_Data, Train_Target, Test_Data, Test_Target, activation):
    IMG_SIZE = 256
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    input_layer = layers.Input(shape=(Train_X.shape[1], Train_X.shape[2], Train_X.shape[3]))
    # First convolutional layer with 32 filters and Kernal size 3*3
    x = layers.Conv2D(32, (3, 3), strides=1, padding='same', activation=activation, input_shape=(150, 150, 1))(input_layer)

    # Batch normalization to stabilize and speed up training
    x = layers.BatchNormalization()(x)

    # Apply CBAM to the first Batch Normalization layer output
    x = cbam_block(x)

    # First max pooling layer to reduce spatial dimensions
    x = layers.MaxPool2D((2, 2), strides=2)(x)

    # Apply CBAM to the first MaxPool layer output
    x = cbam_block(x)

    # Second convolutional layer with 64 filters and Kernal size 3*3
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1))(x)

    # Batch normalization to stabilize and speed up training
    x = layers.BatchNormalization()(x)

    # Apply CBAM to the Second Batch Normalization layer output
    x = cbam_block(x)

    # Second max pooling layer to reduce spatial dimensions
    x = layers.MaxPool2D((2, 2), strides=2)(x)

    # Apply CBAM to the Second MaxPool layer output
    x = cbam_block(x)

    # Third convolutional layer with 128 filters and Kernal size 3*3
    x = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1))(x)

    # Batch normalization to stabilize and speed up training
    x = layers.BatchNormalization()(x)

    # Apply CBAM to the third Batch Normalization layer output
    x = cbam_block(x)

    # Third max pooling layer to reduce spatial dimensions
    x = layers.MaxPool2D((2, 2), strides=2)(x)

    # Apply CBAM to the third MaxPool layer output
    x = cbam_block(x)

    # Fourth convolutional layer with 256 filters and Kernal size 3*3
    x = layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1))(x)

    # Batch normalization to stabilize and speed up training
    x = layers.BatchNormalization()(x)

    # Apply CBAM to the fourth Batch Normalization layer output
    x = cbam_block(x)

    # Fourth max pooling layer to reduce spatial dimensions
    x = layers.MaxPool2D((2, 2), strides=2)(x)

    # Apply CBAM to the fourth MaxPool layer output
    x = cbam_block(x)

    # Flatten layer  and add Dropout and  fully connected layers
    x = layers.Flatten()(x)

    # First dense layer with 256 neurons
    x = layers.Dense(units=256, activation='relu')(x)

    # Dropout layer to reduce overfitting
    x = layers.Dropout(0.20)(x)

    #  Output layer with sigmoid activation for classificatio
    output_layer = layers.Dense(units=Train_Target.shape[1], activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

    # Print a summary of the model
    model.summary()
    model.fit(Train_X, Train_Target, epochs=100, steps_per_epoch=100)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = evaluation(Test_Target, pred)
    return Eval, pred
