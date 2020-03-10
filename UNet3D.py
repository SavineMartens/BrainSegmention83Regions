# U-Net architecture for Savine
# author B.Li
import keras
from keras.models import Model
from keras.layers import Input, concatenate,Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Dropout, Dense
import keras.backend as K
# use if there are more than one GPU
from keras.utils.training_utils import multi_gpu_model
from keras.layers.advanced_activations import LeakyReLU
from keras.engine import Layer
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras.layers import Activation


class Softmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis=axis
        super(Softmax, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def compute_output_shape_for(self, input_shape):
        return input_shape


def dice_coef_prob(y_true, y_pred):
    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]
    intersection = K.sum(FG_true * FG_pred)
    union = K.sum(FG_true) + K.sum(FG_pred)
    return K.mean(2. * intersection / union)

def dice_coef_prob_all(y_true, y_pred):
    FG_true = y_true
    FG_pred = y_pred
    intersection = K.sum(FG_true * FG_pred)
    union = K.sum(FG_true) + K.sum(FG_pred)
    return K.mean(2. * intersection / union)


def dice_coef_binary(y_true, y_pred):
    '''binary dice coefficient using only foreground labels'''
    # remove batch dimension
    sq_true = tf.squeeze(y_true, axis=0)
    sq_pred = tf.squeeze(y_pred, axis=0)
    FG_true = sq_true[:, :, :, 1:]
    arg_y_pred = tf.math.argmax(sq_pred, axis=-1)
    # print('FG_pred')
    with tf.Session() as sess:
        print(sess.run(arg_y_pred))
        # print(arg_y_pred.eval())
    # binarize prediction
    # print('arg_max')
    # with tf.Session() as sess:
    #     print(sess.run(bin_FG_pred))
    #     print(bin_FG_pred.eval())
    int_bin_pred = tf.cast(arg_y_pred, tf.uint8)
    new_bin_pred = tf.expand_dims(int_bin_pred, axis=-1)
    print('integer')
    with tf.Session() as sess:
        print(sess.run(int_bin_pred))
        # print(int_bin_pred.eval())
    one_hot_pred = tf.one_hot(new_bin_pred, tf.keras.backend.int_shape(y_pred)[-1], axis=-1)
    one_hot_FG_pred = one_hot_pred[:, :, :, :, 1:]
    one_hot_FG_pred = tf.squeeze(one_hot_FG_pred, axis=3)
    print('one_hot')
    with tf.Session() as sess:
        print(sess.run(one_hot_FG_pred))
        # print(one_hot_FG_pred.eval())
    intersection = K.sum(FG_true * one_hot_FG_pred)
    print('intersection')
    with tf.Session() as sess:
        print(sess.run(intersection))
    union = K.sum(FG_true) + K.sum(one_hot_FG_pred)
    print('union')
    with tf.Session() as sess:
        print(sess.run(union))
    all_dice = 2. * intersection / union
    avg_dice = K.mean(2. * intersection / union)
    print('intersection')
    with tf.Session() as sess:
        print(sess.run(avg_dice))
    return avg_dice


def dice_loss(y_true, y_pred):
    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]
    intersection = K.sum(FG_true * FG_pred) #+ 1
    union = K.sum(FG_true) + K.sum(FG_pred) #+ 1

    return 1 - K.mean(2. * intersection / union)


def dice_loss_all(y_true, y_pred):
    '''used in MainAllLobes.py'''
    FG_true = y_true
    FG_pred = y_pred
    intersection = K.sum(FG_true * FG_pred) #+ 1
    union = K.sum(FG_true) + K.sum(FG_pred) #+ 1

    return 1 - K.mean(2. * intersection / union)


def weighted_dice_loss(y_true, y_pred):
    '''ratio relative to largest foreground area'''
    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]
    # total_volume = K.sum(K.sum(FG_true))
    volume_per_area = tf.reduce_sum(FG_true, axis=[0, 1, 2, 3])
    # print('volume_per_area:', tf.keras.backend.int_shape(volume_per_area))
    max_area = tf.math.argmax(volume_per_area)
    max_volume = volume_per_area[max_area]
    weights = max_volume/volume_per_area  # weight of largest area = 1

    intersection = K.sum(weights * FG_true * FG_pred)
    union = (weights*K.sum(FG_true) + K.sum(FG_pred))

    return 1 - K.mean(2. * intersection / union)

def weighted_dice_loss2(y_true, y_pred):
    '''ratio relative to individual volume area'''
    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]
    # total_volume = K.sum(K.sum(FG_true))
    volume_per_area = tf.reduce_sum(FG_true, axis=[0, 1, 2, 3])
    weights = 1/volume_per_area
    intersection = K.sum(weights * FG_true * FG_pred)
    union = (weights*K.sum(FG_true) + K.sum(FG_pred))

    return 1 - K.mean(2. * intersection / union)


def weighted_dice_loss3(y_true, y_pred):
    '''Use ratio relative to background'''
    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]
    volume_per_area = tf.reduce_sum(y_true, axis=[0, 1, 2, 3])
    max_area = 0# tf.math.argmax(volume_per_area)
    # print('largest index is', max_area)
    # with tf.Session() as sess:
    #     print(sess.run(max_area))
    max_volume = volume_per_area[max_area]
    weights = max_volume/volume_per_area[1:]  # weight
    # print('weights', weights)
    # with tf.Session() as sess:
    #     print(sess.run(weights))
    intersection = K.sum(weights * FG_true * FG_pred)
    union = (weights*K.sum(FG_true) + K.sum(FG_pred))

    return 1 - K.mean(2. * intersection / union)


def weighted_dice_loss4(y_true, y_pred):
    '''Use ratio relative to total foreground'''
    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]
    # total_volume = K.sum(K.sum(FG_true))
    volume_per_area = tf.reduce_sum(FG_true, axis=[0, 1, 2, 3])
    # print('volume_per_area:', tf.keras.backend.int_shape(volume_per_area))
    FG_volume = tf.reduce_sum(volume_per_area)
    weights = FG_volume/volume_per_area  # weight of largest area = 1

    intersection = K.sum(weights * FG_true * FG_pred)
    union = (weights*K.sum(FG_true) + K.sum(FG_pred))

    return 1 - K.mean(2. * intersection / union)


def squared_weighted_dice_loss(y_true, y_pred):
    #https://arxiv.org/pdf/1707.03237.pdf
    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]
    # total_volume = K.sum(K.sum(FG_true))
    volume_per_area = tf.reduce_sum(FG_true, axis=[0, 1, 2, 3])
    weights = 1.0 / tf.math.square(volume_per_area)
    intersection = K.sum(weights * FG_true * FG_pred)
    union = (weights*K.sum(FG_true) + K.sum(FG_pred))

    return 1 - K.mean(2. * intersection / union)


def bin_log_dice_loss(y_true, y_pred):
    # bin_loss = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    bin_loss = keras.losses.binary_crossentropy(y_true, y_pred)
    return bin_loss + dice_loss(y_true, y_pred)


def focal_loss(y_true, y_pred):
    gamma = float(1)  # the higher the more focus on difficult classes
    alpha = float(1) #float(4) Prior knowledge about class imbalance
    # https://github.com/maozezhong/focal_loss_multi_class/blob/master/focal_loss.py
    # https://arxiv.org/pdf/1708.02002.pdf
    # https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    # below is from: https://github.com/zhezh/focalloss/blob/master/focalloss.py

    model_out = y_pred
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, weight)
    reduced_fl = tf.reduce_max(fl, axis=-1)  # loss per voxel
    sum_red_fl = tf.reduce_sum(reduced_fl, [1, 2, 3])
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max

    return sum_red_fl


def focal_dice_loss(y_true, y_pred):
    gamma = float(1)  # the higher the more focus on difficult classes
    alpha = float(1)  #float(4) Prior knowledge about class imbalance

    FG_true = y_true[:, :, :, :, 1:]
    FG_pred = y_pred[:, :, :, :, 1:]

    weight = alpha*(1 - FG_pred) ** gamma
    intersection = FG_true * FG_pred
    union = FG_true + FG_pred

    loss = K.mean(K.sum(weight * (2. * intersection / union)))

    return loss


# network architecture for multiple classes
def unet_3d_9_BN(img_x, img_y, img_z, img_ch, num_start_channel, num_classes, alpha_relu):

    alpha = alpha_relu

    inputs = Input((img_x, img_y, img_z, img_ch))

    conv11 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(inputs)
    conv11 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv11)
    conv11 = LeakyReLU(alpha=alpha)(conv11)  # small slope to keep activation alive, otherwise neurons can die
    conv12 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv11)
    conv12 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv12)
    conv12 = LeakyReLU(alpha=alpha)(conv12)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv12)

    conv21 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None,use_bias=False, padding='same')(pool1)
    conv21 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv21)
    conv21 = LeakyReLU(alpha=alpha)(conv21)
    conv22 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv21)
    conv22 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv22)
    conv22 = LeakyReLU(alpha=alpha)(conv22)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv22)

    conv31 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(pool2)
    conv31 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv31)
    conv31 = LeakyReLU(alpha=alpha)(conv31)
    conv32 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv31)
    conv32 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv32)
    conv32 = LeakyReLU(alpha=alpha)(conv32)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv32)

    conv41 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False, padding='same')(pool3)
    conv41 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv41)
    conv41 = LeakyReLU(alpha=alpha)(conv41)
    conv42 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False,  padding='same')(conv41)
    conv42 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv42)
    conv42 = LeakyReLU(alpha=alpha)(conv42)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv42)

    conv51 = Conv3D(num_start_channel*16, (3, 3, 3), activation=None, use_bias=False,padding='same')(pool4)
    conv51 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv51)
    conv51 = LeakyReLU(alpha=alpha)(conv51)
    conv52 = Conv3D(num_start_channel*16, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv51)
    conv52 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv52)
    conv52 = LeakyReLU(alpha=alpha)(conv52)


    up6 = UpSampling3D(size=(2, 2, 2))(conv52)

    up6 = concatenate([up6, conv42], axis=4)
    conv61 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False, padding='same')(up6)
    conv61 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv61)
    conv61 = LeakyReLU(alpha=alpha)(conv61)
    conv62 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv61)
    conv62 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv62)
    conv62 = LeakyReLU(alpha=alpha)(conv62)

    up7 = UpSampling3D(size=(2, 2, 2))(conv62)

    up7 = concatenate([up7, conv32], axis=4)
    conv71 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(up7)
    conv71 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv71)
    conv71 = LeakyReLU(alpha=alpha)(conv71)
    conv72 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv71)
    conv72 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv72)
    conv72 = LeakyReLU(alpha=alpha)(conv72)


    up8 = UpSampling3D(size=(2, 2, 2))(conv72)

    up8 = concatenate([up8, conv22], axis=4)
    conv81 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None, use_bias=False, padding='same')(up8)
    conv81 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv81)
    conv81 = LeakyReLU(alpha=alpha)(conv81)
    conv82 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv81)
    conv82 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv82)
    conv82 = LeakyReLU(alpha=alpha)(conv82)

    up9 = UpSampling3D(size=(2, 2, 2))(conv82)

    up9 = concatenate([up9, conv12], axis=4)
    conv91 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(up9)
    conv91 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv91)
    conv91 = LeakyReLU(alpha=alpha)(conv91)
    conv92 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv91)
    conv92 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv92)
    conv92 = LeakyReLU(alpha=alpha)(conv92)
    conv93 = Conv3D(num_classes, (1, 1, 1), activation=None, use_bias=False)(conv92)  #num_classes
    conv93 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv93)

    #conv93 = Activation(K.softmax)(conv93)
    softmax = Softmax()(conv93)


    # model = Model(inputs, softmax)
    model = Model(inputs, softmax)

    return model


class DenseWeights(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DenseWeights, self).__init__(**kwargs)

    def build(self):
        self.kernel = self.add_weight(name='kernel', shape=self.output_dim, initializer='uniform', trainable=True)
        super(DenseWeights, self).build()

    def call(self):
        return self.kernel

    def compute_output_shape(self):
        return self.output_dim





def unet2(img_x, img_y, img_z, img_ch, num_start_channel, num_classes, alpha_relu):

    alpha = alpha_relu

    inputs = Input((img_x, img_y, img_z, img_ch))

    conv11 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(inputs)
    conv11 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv11)
    conv11 = LeakyReLU(alpha=alpha)(conv11)  # small slope to keep activation alive, otherwise neurons can die
    conv12 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv11)
    conv12 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv12)
    conv12 = LeakyReLU(alpha=alpha)(conv12)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv12)

    conv21 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None,use_bias=False, padding='same')(pool1)
    conv21 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv21)
    conv21 = LeakyReLU(alpha=alpha)(conv21)
    conv22 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv21)
    conv22 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv22)
    conv22 = LeakyReLU(alpha=alpha)(conv22)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv22)

    conv31 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(pool2)
    conv31 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv31)
    conv31 = LeakyReLU(alpha=alpha)(conv31)
    conv32 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv31)
    conv32 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv32)
    conv32 = LeakyReLU(alpha=alpha)(conv32)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv32)

    conv41 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False, padding='same')(pool3)
    conv41 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv41)
    conv41 = LeakyReLU(alpha=alpha)(conv41)
    conv42 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False,  padding='same')(conv41)
    conv42 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv42)
    conv42 = LeakyReLU(alpha=alpha)(conv42)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv42)

    conv51 = Conv3D(num_start_channel*16, (3, 3, 3), activation=None, use_bias=False,padding='same')(pool4)
    conv51 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv51)
    conv51 = LeakyReLU(alpha=alpha)(conv51)
    conv52 = Conv3D(num_start_channel*16, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv51)
    conv52 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv52)
    conv52 = LeakyReLU(alpha=alpha)(conv52)


    up6 = UpSampling3D(size=(2, 2, 2))(conv52)

    up6 = concatenate([up6, conv42], axis=4)
    conv61 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False, padding='same')(up6)
    conv61 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv61)
    conv61 = LeakyReLU(alpha=alpha)(conv61)
    conv62 = Conv3D(num_start_channel*8, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv61)
    conv62 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv62)
    conv62 = LeakyReLU(alpha=alpha)(conv62)

    up7 = UpSampling3D(size=(2, 2, 2))(conv62)

    up7 = concatenate([up7, conv32], axis=4)
    conv71 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(up7)
    conv71 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv71)
    conv71 = LeakyReLU(alpha=alpha)(conv71)
    conv72 = Conv3D(num_start_channel*4, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv71)
    conv72 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv72)
    conv72 = LeakyReLU(alpha=alpha)(conv72)


    up8 = UpSampling3D(size=(2, 2, 2))(conv72)

    up8 = concatenate([up8, conv22], axis=4)
    conv81 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None, use_bias=False, padding='same')(up8)
    conv81 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv81)
    conv81 = LeakyReLU(alpha=alpha)(conv81)
    conv82 = Conv3D(num_start_channel*2, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv81)
    conv82 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv82)
    conv82 = LeakyReLU(alpha=alpha)(conv82)

    up9 = UpSampling3D(size=(2, 2, 2))(conv82)

    up9 = concatenate([up9, conv12], axis=4)
    conv91 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(up9)
    conv91 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv91)
    conv91 = LeakyReLU(alpha=alpha)(conv91)
    conv92 = Conv3D(num_start_channel, (3, 3, 3), activation=None, use_bias=False, padding='same')(conv91)
    conv92 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv92)
    conv92 = LeakyReLU(alpha=alpha)(conv92)
    conv93 = Conv3D(num_classes, (1, 1, 1), activation=None, use_bias=False)(conv92)  #num_classes
    conv93 = BatchNormalization(epsilon=0.001, weights=None, momentum=0.9)(conv93)

    #conv93 = Activation(K.softmax)(conv93)
    x = Dense(num_classes, activation='softmax')(conv93)
    softmax = Softmax()(conv93)
    flat_soft = tf.keras.layers.Flatten(softmax)
    x_soft = concatenate([flat_soft, x], axis=-1)
    print(tf.keras.backend.int_shape(x_soft))

    # model = Model(inputs, softmax)
    model = Model(inputs, x_soft)

    return model


def learn_weight_dice_loss(y_true, x_soft):
    num_classes = tf.keras.backend.int_shape(y_true)[-1] - 1
    weights = x_soft[-num_classes, -1]
    FG_true = y_true[:, :, :, :, 1:]
    pred_flatten = x_soft[0:-num_classes]
    y_pred = tf.reshape(pred_flatten, tf.keras.backend.int_shape(y_true))
    FG_pred = y_pred[:, :, :, :, 1:]

    total = []
    for c in range(num_classes):
        true_c = FG_true[:, :, :, :, c]
        pred_c = FG_pred[:, :, :, :, c]
        intersection = weights[c]*K.sum(true_c * pred_c)
        union = (weights[c]*K.sum(FG_true) + K.sum(FG_pred))
        dice = (2*intersection)/ union
        total.append(dice)
    return 1 - K.mean(total)



