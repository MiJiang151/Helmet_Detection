

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

def ARDNet(img_input,classes_num):
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



    def squeeze_excite_block(input, ratio=4):
        init = input
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
        filters = init.shape[channel_axis]  # infer input number of filters
        se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)  # determine Dense matrix shape

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        x = multiply([init, se])
        return x

    def spaceSE(input):
        init = input
        s_ap = Lambda(lambda x:K.mean(x, axis=-1, keepdims=True))(init)
        s_mp = Lambda(lambda x:K.max(x, axis=-1, keepdims=True))(init)
        x = concatenate([s_ap, s_mp], axis=-1)

        s = conv(x,1,(5,5))
        #s = dilationconv(x, 1, (3,3))
        #s = BatchNormalization(momentum=0.9, epsilon=1e-5)(s)
        s = Activation('sigmoid')(s)
        x = multiply([init, s])
        return x

    def eca(input, b=1, gamma=2,):
        channel = input.shape[-1]
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        avg_pool = GlobalAveragePooling2D()(input)

        x = Reshape((-1, 1))(avg_pool)
        x = Conv1D(1, kernel_size=kernel_size, padding="same",  use_bias=False, )(x)
        x = Activation('sigmoid')(x)
        x = Reshape((1, 1, -1))(x)

        output = multiply([input, x])
        return output

    def channel_attention(input_feature, ratio=8):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature._shape_val[channel_axis]

        shared_layer_one = Dense(channel // ratio,
                                 kernel_initializer='he_normal',
                                 activation='relu',
                                 use_bias=True,
                                 bias_initializer='zeros')

        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        assert avg_pool._shape_val[1:] == (1, 1, channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool._shape_val[1:] == (1, 1, channel // ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool._shape_val[1:] == (1, 1, channel)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        assert max_pool._shape_val[1:] == (1, 1, channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool._shape_val[1:] == (1, 1, channel // ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool._shape_val[1:] == (1, 1, channel)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('hard_sigmoid')(cbam_feature)

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def spatial_attention(input_feature):
        kernel_size = 7
        if K.image_data_format() == "channels_first":
            channel = input_feature._shape_val[1]
            cbam_feature = Permute((2, 3, 1))(input_feature)
        else:
            channel = input_feature._shape_val[-1]
            cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool._shape_val[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool._shape_val[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        assert concat._shape_val[-1] == 2
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              activation='hard_sigmoid',
                              strides=1,
                              padding='same',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        assert cbam_feature._shape_val[-1] == 1

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def cbam_block(cbam_feature, ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in CBAM: Convolutional Block Attention Module.
        """

        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature, )
        return cbam_feature

    def GAM(input,rate=4,):
            # channel attention
            init = input
            channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
            filters = init.shape[channel_axis]  # infer input number of filters
            gam_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)  # determine Dense matrix shape

            tmp = GlobalAveragePooling2D()(init)
            tmp = Reshape(gam_shape)(tmp)
            tmp = Dense(int(filters / rate), activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(tmp)

            mc = Dense(int(filters),activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(tmp)

            f2 = mc * input

            # spatial attention
            tmp = Conv2D(filters=24, kernel_size=7, padding='same')(f2)
            tmp = BatchNormalization()(tmp)
            tmp = Activation('relu')(tmp)
            tmp = Conv2D(filters=24, kernel_size=7, padding='same')(tmp)
            ms = BatchNormalization()(tmp)
            x = ms * f2

            return x

    def gam1(input_shape, rate=4):
        init = input_shape
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
        filters = init.shape[channel_axis]  # infer input number of filters
        gam_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)  # determine Dense matrix shape


        # channel attention
        tmp = Reshape((-1, int(filters)))(init)
        tmp = Dense(int(filters / rate))(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Dense(int(filters))(tmp)
        mc = Reshape(gam_shape[1:])(tmp)
        f2 = mc * init

        # spatial attention
        tmp = Conv2D(int(filters / rate), kernel_size=7, padding='same')(f2)
        tmp = BatchNormalization(tmp)
        tmp = Activation('relu')(tmp)
        tmp = Conv2D(int(filters), kernel_size=7, padding='same')(tmp)
        ms = BatchNormalization(tmp)
        f3 = ms * f2

        return f3
    # （1）通道注意力
    def channel_attention(inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 重新排序维度[b,h,w,c]==>[b,c,h,w]
        x = tf.transpose(inputs, perm=[0, 3, 1, 2])  # perm代表重新排序的轴
        # 重塑特征图尺寸[b,c,h,w]==>[b,c,h*w]
        x_reshape = tf.reshape(x, shape=[-1, c, h * w])

        # 重新排序维度[b,c,h*w]==>[b,h*w,c]
        x_reshape_trans = tf.transpose(x_reshape, perm=[0, 2, 1])  # 指定需要交换的轴
        # 矩阵相乘
        x_mutmul = x_reshape_trans @ x_reshape
        # 经过softmax归一化权重
        x_mutmul = tf.nn.softmax(x_mutmul)

        # reshape后的特征图与归一化权重矩阵相乘[b,x,h*w]
        x = x_reshape @ x_mutmul
        # 重塑形状[b,c,h*w]==>[b,c,h,w]
        x = tf.reshape(x, shape=[-1, c, h, w])
        # 重新排序维度[b,c,h,w]==>[b,h,w,c]
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x = x * gama

        # 输入和输出特征图叠加
        x = layers.add([x, inputs])

        return x

    # （2）位置注意力
    def position_attention(inputs):
        # 定义可训练变量，反向传播可更新
        gama = tf.Variable(tf.ones(1))  # 初始化1

        # 获取输入特征图的shape
        b, h, w, c = inputs.shape

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c//8]
        x1 = layers.SeparableConv2D(filters=c // 8, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c//8]==>[b,c//8,h,w]
        x1_trans = tf.transpose(x1, perm=[0, 3, 1, 2])
        # 重塑特征图尺寸[b,c//8,h,w]==>[b,c//8,h*w]
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c // 8, h * w])
        # 调整维度排序[b,c//8,h*w]==>[b,h*w,c//8]
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape, perm=[0, 2, 1])
        # 矩阵相乘
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        # 经过softmax归一化权重
        x1_mutmul = tf.nn.softmax(x1_mutmul)

        # 深度可分离卷积[b,h,w,c]==>[b,h,w,c]
        x2 = layers.SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1, padding='same')(inputs)
        # 调整维度排序[b,h,w,c]==>[b,c,h,w]
        x2_trans = tf.transpose(x2, perm=[0, 3, 1, 2])
        # 重塑尺寸[b,c,h,w]==>[b,c,h*w]
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c, h * w])

        # 调整x1_mutmul的轴，和x2矩阵相乘
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2, 1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans

        # 重塑尺寸[b,c,h*w]==>[b,c,h,w]
        x2_mutmul = tf.reshape(x2_mutmul, shape=[-1, c, h, w])
        # 轴变换[b,c,h,w]==>[b,h,w,c]
        x2_mutmul = tf.transpose(x2_mutmul, perm=[0, 2, 3, 1])
        # 结果乘以可训练变量
        x2_mutmul = x2_mutmul * gama

        # 输入和输出叠加
        x = layers.add([x2_mutmul, inputs])
        return x

    # （3）DANet网络架构
    def danet(inputs):
        # 输入分为两个分支
        x1 = channel_attention(inputs)  # 通道注意力
        x2 = position_attention(inputs)  # 位置注意力

        # 叠加两个注意力的结果
        x = layers.add([x1, x2])
        return x

    def CA_block(input,w=64,h=64):
            residual = input
            s = residual
            # n, c, h, w = x.shape

            x_h = AvgPool2D(pool_size=(1, w), strides=1, padding='same')(s)
            x_w = AvgPool2D(pool_size=(h, 1), strides=1, padding='same')(s)

            x_w = transpose(x_w, [0, 2, 1, 3])

            x = concat([x_h, x_w], axis=1)
            x = Conv2D(filters=8, kernel_size=(1, 1), strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('sigmoid')(x)

            x_h, x_w = split(x, 2, axis=1)
            x_w = transpose(x_w, [0, 2, 1, 3])

            x_h = Conv2D(filters=24, kernel_size=(1, 1), strides=1, padding='same')(x_h)
            x_w = Conv2D(filters=24, kernel_size=(1, 1), strides=1, padding='same')(x_w)
            x_h = Activation('sigmoid')(x_h)
            x_w = Activation('sigmoid')(x_w)
            x = residual * x_w * x_h
            print(np.shape(x))
            return x

    def bottleneck(x,increase=False):
        channels = nchannels
        x10 = conv(x, growth_rate, (1,1))
        x10 = bn_relu(x10)
        x10 = conv(x10, growth_rate, (1,1))
        x11 = bn_relu(x10)
        x12 = conv(x11,growth_rate, (3,3))
        x13 = bn_relu(x12)
        x13 = CA_block(x13)
        x13 = spaceSE(x13)
        x21 = conv(x13, channels, (1,1))
        x22 = bn_relu(x21)
        x23 = conv(x22,growth_rate, (3,3))
        x32 = bn_relu(x23)

        block = add([x32, x11])
        return block

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1,1))
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def ARD(x,blocks,nchannels):
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
    x, nchannels = ARD(x,3,nchannels)   #32*32*(3*24+48)=64*64*120
    x, nchannels = transition(x,nchannels)           #16*16*60
    x, nchannels = ARD(x,6,nchannels)   #16*16*(6*24+60)=16*16*204
    x, nchannels = transition(x,nchannels)     #8*8*102
    x, nchannels = ARD(x,9,nchannels)  #8*8*(12*24+102)=8*8*390
    x, nchannels = transition(x,nchannels)    #4*4*185
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)

    return x