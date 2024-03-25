# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:23:37 2020

@author: Administrator
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import Input
from dataset import DataSet
from model import CPAGNet
from matplotlib import pyplot as plt


from tensorflow.keras import backend as K
if('tensorflow' == K.backend()):
    #from keras.backend.tensorflow_backend import set_session
    config = tf.compat.v1.ConfigProto()

    #config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

#定义可变学习率
def scheduler(epoch):
    if epoch < 20:
        return 0.1
    if epoch < 30:
        return 0.01
    if epoch < 40:
        return 0.001
    return 0.0001

#加载数据集
X_train,Y_train,X_test,Y_test = DataSet()

log_filepath = 'E:/aa/Seatbelt-detection11.8/Seatbelt-detection/logs/TEST1logs'#日志文件保存地址/
img_input = Input(shape=(64,64,3))
output    = CPAGNet(img_input,2)  # 5
resnet    = Model(img_input, output)

num_classes        = 2
batch_size         = 64
epochs             = 50


#设置优化器参数
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
resnet.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#日志文件所需保存信息
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)

#自适应学习率
change_lr = LearningRateScheduler(scheduler)

cbks = [change_lr,tb_cb]

resnet.fit(X_train, Y_train,batch_size=batch_size,
                     #steps_per_epoch=iterations,
                     epochs=epochs,
                     #callbacks=[cbks,checkpoint],
                     callbacks=cbks,
                     validation_data=(X_test, Y_test)
                     )



#模型权重保存
resnet.save('./checkpoint/' + 'TEST1.h5')


"""
model = CPAGNet()     model==resnet
history = model.fit(X_train, Y_train, batch_size=10, epochs=5,
                    validation_data=(X_test, Y_test),
                    validation_freq=1,
                    callbacks=[cp_callback])
############################  show  #############################
# 显示训练集和验证集的acc和loss曲线
acc=history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
"""

