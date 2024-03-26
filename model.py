# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:19:31 2021

@author: Administrator
"""
import numpy as np
from tensorflow.keras.layers import AvgPool2D,Permute,Layer,Conv2D, BatchNormalization, Activation
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid
from tensorflow import split, concat, transpose
from CA import Coordinate_Attention
import tensorflow as tf
import math
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.keras.layers import Conv2D,Conv1D,Dense,GlobalMaxPooling2D,add,Concatenate,Add,Activation,multiply, AveragePooling2D, GlobalAveragePooling2D, Lambda, concatenate,Reshape
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
if('tensorflow' == K.backend()):
    #from keras.backend.tensorflow_backend import set_session
    config = tf.compat.v1.ConfigProto()

    #config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


growth_rate        = 24#每层密集连接单元增加的通道数
compression        = 0.5
weight_decay       = 0

def CPAGNet(img_input,classes_num):
    def conv(x, out_filters, k_size):
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    def dense_layer(x):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x,increase=False):
        channels = nchannels

        x10 = conv(x, growth_rate, (3,3))
        x10 = bn_relu(x10)
        x10 = conv(x10, growth_rate, (3,3))
        x11 = bn_relu(x10)

        #spaceSE   eca   cbam_block
        x18 = CA_block(x11)
        x19 = spaceSE(x18)

        x12 = conv(x19, channels, (3,3))
        x13 = bn_relu(x12)
        x16 = conv(x13, channels, (3,3))
        x17 = bn_relu(x16)
        x23 = conv(x17,growth_rate, (3,3))
        x32 = bn_relu(x23)
        block = add([x32, x11,x19])
        return block

    def Integration(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1,1))
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def CPAG(x,blocks,nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat,nchannels)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    nchannels = growth_rate*2
    #x=64*64*3
    x = conv(img_input, nchannels, (3,3))
    x = AveragePooling2D((3,3), strides=(2, 2))(x)  #32*32*48
    x, nchannels = CPAG(x,3,nchannels)   #32*32*(3*24+48)=64*64*120
    x, nchannels = Integration(x,nchannels)           #16*16*60
    x, nchannels = CPAG(x,6,nchannels)   #16*16*(6*24+60)=16*16*204
    x, nchannels = Integration(x,nchannels)     #8*8*102
    x, nchannels = CPAG(x,12,nchannels)  #8*8*(12*24+102)=8*8*390
   # x, nchannels = Integration(x,nchannels)    #4*4*185
   # x, nchannels = CPAG(x,18,nchannels)  #8*8*(12*24+102)=8*8*390
    #x= conv()(x)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x
