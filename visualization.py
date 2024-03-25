# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:23:27 2021

@author: Administrator
"""
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np

# 加载训练好的模型
model = load_model(r"./checkpoint/ARD-Net.h5")

model.summary()  # 打印模型.查看参数及层的名称



images=cv2.imread("./img/look.jpg")

    # 根据载入的训练好的模型的配置，将图像统一尺寸
image_arr = cv2.resize(images, (64, 64))
image_arr = image_arr.astype('float')/255.0

image_arr = np.expand_dims(image_arr, axis=0)

# 第一个 model.layers[0],不修改,表示输入数据；
# 第二个model.layers[ ],修改为需要输出的层数的编号[]
layer_1 = K.function([model.layers[0].input], [model.layers[16].output])

# 只修改inpu_image
f1 = layer_1([image_arr])[0]
f2 = np.sum(f1, axis=-1)
    # 第一层卷积后的特征图展示，输出是（1,66,66,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
plt.figure()
label_tmp = f2[0,:,:]
plt.imshow(f2[0,:,:],cmap="viridis") # cmap='gray' 显示出什么颜色

#plt.imshow(label_tmp.reshape(31,31))
# 去除坐标轴
plt.xticks([])
plt.yticks([])
plt.savefig(r".img/visualization.png")
plt.show()
