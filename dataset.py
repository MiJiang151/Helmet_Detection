# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:37:22 2020

@author: Administrator
"""
import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import random

def DataSet():
    # 首先需要定义训练集和测试集的路径，这里创建了 train ， 和 test 文件夹
    # 每个文件夹下又创建了 glue，medicine 两个文件夹，所以这里一共四个路径
    train_path_dog ='./dataset/train/no/'
    train_path_cat = './dataset/train//yes/'
    
    test_path_dog ='./dataset/test/no/'
    test_path_cat = './dataset/test/yes/'
    
    # os.listdir(path) 是 python 中的函数，它会列出 path 下的所有文件名
    # 比如说 imglist_train_glue 对象就包括了/train/glue/ 路径下所有的图片文件名
    imglist_train_dog = os.listdir(train_path_dog)
    imglist_train_cat = os.listdir(train_path_cat)
    
    # 下面两行代码读取了 /test/dog 和 /test/cat 下的所有图片文件名
    imglist_test_dog = os.listdir(test_path_dog)
    imglist_test_cat = os.listdir(test_path_cat)
    
    # 这里定义两个 numpy 对象，X_train 和 Y_train
    
    # X_train 对象用来存放训练集的图片。每张图片都需要转换成 numpy 向量形式
    # X_train 的 shape 是 (22，64，64，3) 
    # 22 是训练集中图片的数量（训练集中固体胶和西瓜霜图片数量之和）
    # 因为 模型结构 要求输入的图片尺寸是 (64,64) , 所以要设置成相同大小（也可以设置成其它大小，参看 keras 的文档）
    # 3 是图片的通道数（rgb）
    
    # Y_train 用来存放训练集中每张图片对应的标签
    # Y_train 的 shape 是 （22，2）
    # 22 是训练集中图片的数量（训练集中固体胶和西瓜霜图片数量之和）
    # 因为一共有两种图片，所以第二个维度设置为 2
    # Y_train 大概是这样的数据 [[0,1],[0,1],[1,0],[0,1],...]
    # [0,1] 就是一张图片的标签，这里设置 [1,0] 代表 狗，[0,1] 代表猫
    # 如果你有三类图片 Y_train 就因该设置为 (your_train_size,3)
    
    X_train = np.empty((len(imglist_train_dog) + len(imglist_train_cat), 64, 64,3))
    Y_train = np.empty((len(imglist_train_dog) + len(imglist_train_cat), 2))
    # count 对象用来计数，每添加一张图片便加 1
    count = 0
    # 遍历 /train/dog 下所有图片，即训练集下所有的狗图片
    for img_name in imglist_train_dog:
        # 得到图片的路径
        img_path = train_path_dog + img_name
        # 通过 image.load_img() 函数读取对应的图片，并转换成目标大小
        #  image 是 tensorflow.keras.preprocessing 中的一个对象
        img = image.load_img(img_path, target_size=(64, 64))
        # 将图片转换成 numpy 数组，并除以 255 ，归一化
        # 转换之后 img 的 shape 是 （64，64，3）
        img = image.img_to_array(img) / 255.0
        
        # 将处理好的图片装进定义好的 X_train 对象中
        X_train[count] = img
        # 将对应的标签装进 Y_train 对象中，这里都是 狗（dog）图片，所以标签设为 [1,0]
        Y_train[count] = np.array((1,0))
        count+=1
    # 遍历 /train/cat 下所有图片，即训练集下所有的猫图片
    for img_name in imglist_train_cat:

        img_path = train_path_cat + img_name
        img = image.load_img(img_path, target_size=(64,64))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1))
        count+=1
        
    # 下面的代码是准备测试集的数据，与上面的内容完全相同，这里不再赘述
    X_test = np.empty((len(imglist_test_dog) + len(imglist_test_cat), 64, 64,3))
    Y_test = np.empty((len(imglist_test_dog) + len(imglist_test_cat), 2))
    count = 0
    for img_name in imglist_test_dog:

        img_path = test_path_dog + img_name
        img = image.load_img(img_path, target_size=(64, 64))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((1,0))
        count+=1
        
    for img_name in imglist_test_cat:
        
        img_path = test_path_cat + img_name
        img = image.load_img(img_path, target_size=(64, 64))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,1))
        count+=1
        
	# 打乱训练集中的数据
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    # 打乱测试集中的数据
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test = Y_test[index]	

    return X_train,Y_train,X_test,Y_test
