from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf





#from dataset import DataSet
#加载模型h5文件
from matplotlib import pyplot as plt
#import csv
from dataset import DataSet

def PR_curve(y,pred):
    pos = np.sum(y == 1)
    #neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]  # 从大到小排序
    index = np.argsort(pred)[::-1]  # 从大到小排序
    y_sort = y[index]
    #print(y_sort)

    Pre = []
    Rec = []
    thr = []
    for i, item in enumerate(pred_sort):
        if i == 0:#因为计算precision的时候分母要用到i，当i为0时会出错，所以单独列出
            Pre.append(1)
            Rec.append(0)
            thr.append(item)

        else:
            Pre.append(np.sum((y_sort[:i] == 1)) /i)
            Rec.append(np.sum((y_sort[:i] == 1)) / pos)
            thr.append(item)
    return Pre, Rec, thr

def Get_ROC(y_true,pos_prob):
    pos = y_true[y_true==1]#正样本个数
    neg = y_true[y_true==0]#负样本个数
    threshold = np.sort(pos_prob)[::-1]        # 按概率大小逆序排列
    y = y_true[pos_prob.argsort()[::-1]]  #
    tpr_all = [0] ; fpr_all = [0]
    tpr = 0 ; fpr = 0
    x_step = 1/float(len(neg))
    y_step = 1/float(len(pos))
    y_sum = 0                                  # 用于计算AUC
    for i in range(len(threshold)):
        if y[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr
    return tpr_all,fpr_all,y_sum*x_step,threshold       # 获得总体TPR，FPR和相应的AUC


def ACC_TPR_FPR_RRE_F1_KAPPA(y_true,y_pred):#1为正样本，0为负样本，1、0表示在真实onrhot中1所在位置索引值
   # true positive
   TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
   # false positive
   FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))

   # true negative
   TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

   # false negative
   FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))

   Precision = TP/(TP + FP)
   Recall = TP/(TP + FN)

   Accuracy = (TP + TN)/(TP + FP + TN + FN)
   FPR = FP/(FP + TN)
   F1_score = 2*Precision*Recall/(Precision + Recall)

   Confus_matrix = np.array([[TP, FP], [FN, TN]])


   return Precision, Recall, Accuracy, F1_score, FPR, Confus_matrix

def writer_cvs(a,b,c):
    friendInfo=[]
    #将三个列表合并，并创建一个新的列表
    for i in range(len(a)):
        for j in range(len(b)):
            if i==j:
                for k in range(len(c)):
                    if j==k:
                        t=(a[i],b[j],c[k])
                        friendInfo.append(t)
    return friendInfo
_,_,X_test,Y_test = DataSet()
model = load_model("./checkpoint/CAPAqupool-95.41-Net.h5")
#print("X_test",X_test )

pred_y = model.predict(X_test)#预测验证集图像
#print("pred_y",pred_y )
predlabel = pred_y[:,1]#切片切第二维度信息，01标签为预测佩戴头盔
#print("predlabel",predlabel )
predlabel1 =np.round(predlabel, 0)#取整数
print("predlabel1",predlabel1 )
truelabel = Y_test[:,1]#真实标签切片01为佩戴头盔
print("truelabel",truelabel )

Precision,Recall,Accuracy,F1_score,FPR,Confus_matrix = ACC_TPR_FPR_RRE_F1_KAPPA(truelabel,predlabel1)
print("混淆矩阵",Confus_matrix)
print("精确率为:", Precision)
print("召回率为:", Recall)
print("总体精度为:", Accuracy)
print("F1分数为:", F1_score)
print("FPR假阳性为:", FPR)



Pre, Rec, thr = PR_curve(truelabel,predlabel)#完全显示小数位


'''
#输出为文件
P_R = writer_cvs(Pre, Rec, thr)
with open(r"Pre_Rec_thr.csv", 'w+', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(P_R)
    #print(Pre)
    #print(Rec)
    #print(thr)
'''

#绘制PR曲线
plt.plot(Rec, Pre, 'k')
plt.title('Receiver Operating Characteristic')
plt.plot([(0, 0), (1, 1)], 'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 01.01])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

