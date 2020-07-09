# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn import svm

from app.main.feature import load_test
from sklearn.externals import joblib
import os

PATH = os.path.abspath(os.path.dirname(__file__))



def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x

if __name__ == '__main__':
    m = 0
    data, label = load_test()
    for data1 in data:
        print(len(data1))
        # data1 = data1[:800]
        scaler = joblib.load(PATH + "/Model/scaler2.m")
        data1 = scaler.transform(data1)
        data1 = data1[:800]
        pro = 0
        clf_1 = joblib.load("./Model/svm5.m")
        clf_2 = joblib.load("./Model/svm6.m")
        clf_3 = joblib.load("./Model/svm7.m")
        clf_4 = joblib.load("./Model/svm8.m")
        reult1 = clf_1.predict(data1)
        # print(np.sum(reult1 == np.argmax(np.bincount(reult1))))
        r = 0
        for i in clf_1.predict_proba(data1):
            if max(i) < 0.9:
                r = r + 1
        # print(r)
        # print(np.argmax(np.bincount(reult1)))
        pro = pro + np.sum(reult1 == np.argmax(np.bincount(reult1)))
        # print(clf_1.predict_proba(data1))

        reult2 = clf_2.predict(data1)
        # print(np.sum(reult2 == np.argmax(np.bincount(reult2))))
        pro = pro + np.sum(reult2 == np.argmax(np.bincount(reult2)))

        reult3 = clf_3.predict(data1)
        # print(np.sum(reult3 == np.argmax(np.bincount(reult3))))
        pro = pro + np.sum(reult3 == np.argmax(np.bincount(reult3)))

        reult4 = clf_4.predict(data1)
        # print(np.sum(reult4 == np.argmax(np.bincount(reult4))))
        pro = pro + np.sum(reult4 == np.argmax(np.bincount(reult4)))
        pro = pro / 4
        # print(pro)

        res1 = set(reult1)
        res2 = set(reult2)
        res3 = set(reult3)
        res4 = set(reult4)
        # print(res1)
        # print(reult2)
        # print(reult3)
        # print(reult4)

        if len(res1) == 1 & len(res2) == 1 & len(res3) == 1 & len(res4) == 1:
            if res1 == res2 == res3 == res4:
                print(list(res1)[0])
        else:
            train2_y = np.hstack((reult1, reult2, reult3, reult4))
            train2 = np.row_stack((data1, data1, data1, data1))
            # print(train2.shape)
            # print(train2_y.shape)

            # train2 = np.array(train2)
            train2_y = np.array(train2_y)
            clf = svm.SVC(kernel='rbf', C=100, gamma=0.01, probability=True)
            clf.fit(train2, train2_y)
            res = clf.predict(data1)
            print(np.argmax(np.bincount(res)))
            print(np.sum(res == np.argmax(np.bincount(res))))
            b = np.sum(res == np.argmax(np.bincount(res)))
            # y_test = np.ones(len(data1)) * b
            # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
            y_score = clf.predict_proba(data1)
            # print(y_score)
            s = y_score.shape
            y_one_hot = label_binarize(res, np.arange(s[1]))  # 装换成类似二进制的编码
            print(y_one_hot)
            print(res)


            # 1、调用函数计算micro类型的AUC
            # 2、手动计算micro类型的AUC
            # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
            fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
            print(fpr,tpr)
            auc = metrics.auc(fpr, tpr)
            print(auc)
            # 绘图
            mpl.rcParams['font.sans-serif'] = u'SimHei'
            mpl.rcParams['axes.unicode_minus'] = False
            # FPR就是横坐标,TPR就是纵坐标
            plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
            plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
            plt.xlim((-0.01, 1.02))
            plt.ylim((-0.01, 1.02))
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.xlabel('False Positive Rate', fontsize=13)
            plt.ylabel('True Positive Rate', fontsize=13)
            plt.grid(b=True, ls=':')
            plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
            plt.title(u'ROC和AUC', fontsize=17)
            plt.show()
        print(label[m])
        m += 1



