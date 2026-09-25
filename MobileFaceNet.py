import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras import regularizers, Model
from tensorflow.keras.layers import Layer, Input, Conv2D, DepthwiseConv2D, ZeroPadding2D, BatchNormalization, ReLU, Add, \
    Flatten
import math
from Evaluation import Reg_evaluation


class ArcFace(Layer):
    def __init__(self, input_shape, n_classes=7, s=64.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)
        super(ArcFace, self).build(self.input_spec)

    def call(self, inputs):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)

        embedding, labels = inputs

        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = tf.divide(embedding, embedding_norm, name='norm_embedding')
        weights = self.W
        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.divide(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = self.s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = labels
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')

        logit = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        out = tf.nn.softmax(logit)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        return {'s': self.s, 'm': self.m, 'n_classes': self.n_classes}


def correct_pad(inputs, kernel_size):
    img_dim = 2 if tf.keras.backend.image_data_format() == 'channels_first' else 1
    input_size = tf.keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def inverted_res_block(inputs, expansion, stride, filters, block_id):
    channel_axis = -1
    in_channels = tf.keras.backend.int_shape(inputs)[channel_axis]
    pointwise_filters = filters
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    # Expand
    x = Conv2D(
        expansion * in_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand')(x)
    x = BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand_BN')(x)

    # Depthwise
    if stride == 2:
        x = ZeroPadding2D(
            padding=correct_pad(x, 3),
            name=prefix + 'pad')(x)
    x = DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding='same' if stride == 1 else 'valid',
        name=prefix + 'depthwise')(x)
    x = BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise_BN')(x)

    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'project')(x)
    x = BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])

    return x


def mobilefacenet_arcface(Train_X, n_classes, sol):
    weight_decay = 0.00005
    act = ['linear', 'relu', 'softmax', 'sigmoid', 'tanh']
    # input
    input = Input(shape=(Train_X.shape[1], Train_X.shape[2], 3), name='input')
    y = Input(shape=n_classes)

    # Conv2D
    x = Conv2D(sol[0], (3, 3), padding='same', activation=act[sol[2]], strides=(2, 2), kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(weight_decay), use_bias=False, name='sec1_conv2d')(input)
    x = BatchNormalization(name='sec1_bn')(x)
    x = ReLU(name='sec1_relu')(x)

    # DepthwiseConv2D
    x = DepthwiseConv2D((3, 3), padding='same', strides=(1, 1), kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(weight_decay), use_bias=False, name='sec2_depthwiseconv2d')(
        x)
    x = BatchNormalization(name='sec2_bn')(x)
    x = ReLU(name='sec2_relu')(x)
    x = Conv2D(64, (1, 1), padding='same', activation=act[sol[2]], strides=(1, 1), kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(weight_decay), name='sec2_conv2d', use_bias=False)(x)
    x = BatchNormalization(name='sec2_bn2')(x)

    # InvertedResidualBlock
    x = inverted_res_block(x, 2, 2, 64, 0)
    x = inverted_res_block(x, 4, 2, 128, 1)
    x = inverted_res_block(x, 2, 1, 128, 2)
    x = inverted_res_block(x, 4, 2, 128, 3)
    x = inverted_res_block(x, 2, 1, 128, 4)

    # Conv2D 1x1
    x = Conv2D(512, (1, 1), padding='same', strides=(1, 1), kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(weight_decay), name='sec8_conv2d', use_bias=False)(x)
    x = BatchNormalization(name='sec8_bn')(x)
    x = ReLU(name='sec8_relu')(x)

    # linear GDConv 7x7
    x = DepthwiseConv2D((7, 7), padding='valid', strides=(1, 1), kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(weight_decay), use_bias=False, name='sec9_depthwiseconv2d')(
        x)
    x = BatchNormalization(name='sec9_bn')(x)
    x = Conv2D(512, (1, 1), padding='valid', strides=(1, 1), kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(weight_decay), name='sec9_conv2d', use_bias=False)(x)
    x = BatchNormalization(name='sec9_bn2')(x)

    # linear Conv2D 1x1
    x = Conv2D(128, (1, 1), padding='same', strides=(1, 1), kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(weight_decay), use_bias=False, name='sec10_conv2d')(x)
    x = BatchNormalization(name='sec10_bn')(x)

    # faltten
    x = Flatten(dtype='float32')(x)

    # embedding
    x = tf.keras.layers.Lambda(lambda k: tf.keras.backend.l2_normalize(k, axis=-1), name="embedding")(x)

    # loss function
    output = ArcFace(input, n_classes, dtype='float32')([x, y])

    return Model(input, output)


def Model_MobileFaceNet(Train_Data, Train_Target, Test_Data, Test_Target, epoch, sol=None):
    if sol is None:
        sol = [5, 0.01, 1]
    IMG_SIZE = 256
    Train_X = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    model = mobilefacenet_arcface(Train_X, Train_Target.shape[1], sol)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=sol[1]), metrics=['accuracy'])
    model.fit(Train_X, Train_Target, epochs=epoch, steps_per_epoch=100)
    pred = model.predict(Test_X)
    avg = (np.min(pred) + np.max(pred)) / 2
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    Eval = Reg_evaluation(Test_Target, pred)

    return Eval, pred
