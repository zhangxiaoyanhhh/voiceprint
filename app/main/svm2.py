from sklearn import svm

from app.main.feature import load_test,pre_data
from app.main.model import Session, Tvoice,engine
from sklearn.model_selection import KFold
import numpy as np
from sklearn.externals import joblib
import sklearn.preprocessing as preprocessing
from sklearn import model_selection
from sklearn import metrics
import os
import matplotlib.pyplot as plt
from collections import Counter

PATH = os.path.abspath(os.path.dirname(__file__))


session = Session()
def svm_2():

    train_x = []
    Y = []
    j = 0
    b = session.query(Tvoice).all()
    for i in b:
        x = np.frombuffer(i.feature).reshape(i.qian, i.hou)
        # print(x[200:1000])
        # print(np.isfinite(x[200:1000]).all())

        train_x.append(x[200:1000])

        for m in range(800):
            Y.append(j)
        j = j+1
        # print(len(Y))
        # print(b)


    data = train_x[0]
    for i in range(1,len(train_x)):
        print(i)
        data = np.row_stack((data ,train_x[i]))
    print(data.shape)
    Y = np.array(Y)
    print(Y.shape)
    scaler = preprocessing.StandardScaler()  # 标准化转换
    scaler.fit(data)  # 训练标准化对象
    joblib.dump(scaler, PATH + "/Model/scaler2.m")
    data = scaler.transform(data)  # 转换数据集
    # kernel=['rbf']
    # C=[0.1,1,5,100]
    #
    # parameters = {'kernel':kernel,'C':C}
    # grid_svc = model_selection.GridSearchCV(estimator = svm.SVC(),
    #                                         param_grid =parameters,
    #                                         scoring='accuracy',cv=2,verbose =1)
    # # 模型在训练数据集上的拟合
    # X_train = np.array(data)
    # y_train = np.array(Y)
    # grid_svc.fit(X_train,y_train)
    # 返回交叉验证后的最佳参数值
    # print(grid_svc.best_params_, grid_svc.best_score_)

    clf1 = svm.SVC(C=5, decision_function_shape='ovo',probability=True)
    clf1.fit(data, Y)
    joblib.dump(clf1, PATH+"/Model/svm5.m")
    print("res1")
    # reult1 = clf1.predict(data1)
    # res_1 = []
    # for i in clf1.predict_proba(data1):
    #     if max(i) < 0.8:
    #         res_1.append(-1)
    # res1 = set(reult1)
    # print(len(res_1))
    # print(reult1)

    clf2 = svm.SVC(C=6, decision_function_shape='ovo',probability=True)
    clf2.fit(data, Y)
    joblib.dump(clf2, PATH+"/Model/svm6.m")
    print("res1")
    # reult2 = clf2.predict(data1)
    # res2 = set(reult2)
    # res_2 = []
    # for i in clf2.predict_proba(data1):
    #     if max(i) < 0.9:
    #         res_2.append(-1)
    # print(len(res_2))
    # print(res2)

    clf3 = svm.SVC(C=4, decision_function_shape='ovo',probability=True)
    clf3.fit(data, Y)
    joblib.dump(clf3, PATH+"/Model/svm7.m")
    print("res1")
    # reult3 = clf3.predict(data1)
    # res3 = set(reult3)
    # res_3 = []
    # for i in clf3.predict_proba(data1):
    #     if max(i) < 0.9:
    #         res_3.append(-1)
    # print(len(res_3))
    # print(res3)


    clf4 = svm.SVC(C=5.5, decision_function_shape='ovo',probability=True)
    clf4.fit(data, Y)
    joblib.dump(clf4, PATH+"/Model/svm8.m")
    print("res1")
    # reult4 = clf4.predict(data1)
    # res4 = set(reult4)
    # res_4 = []
    # for i in clf4.predict_proba(data1):
    #     if max(i) < 0.9:
    #         res_4.append(-1)
    # print(len(res_4))
    # print(res4)
    # if len(res1) == 1 & len(res2) == 1 & len(res3) == 1 & len(res4) == 1:
    #     if res1 == res2 == res3 == res4:
    #         print(res1)
    # else:
    #     train2_y = np.hstack((reult1, reult2, reult3, reult4))
    #     train2 = np.row_stack((data1, data1, data1, data1))
    #     # print(train2.shape)
    #     # print(train2_y.shape)
    #
    #     # train2 = np.array(train2)
    #     train2_y = np.array(train2_y)
    #     clf = svm.SVC(kernel='rbf', C=100, gamma=0.01, probability=True)
    #     clf.fit(train2, train2_y)
    #     res = clf.predict(data1)
    #     print(np.argmax(np.bincount(res)))
        # print(res)

def test2():
    m = 0
    data, label = load_test()
    score = []
    name_label = []
    for data1 in data:
        # print(len(data1))
        scaler = joblib.load(PATH + "/Model/scaler2.m")
        data1 = scaler.transform(data1)
        data1 = data1[200:1000]
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
        # score.append(pro)
        print(pro)

        res1 = set(reult1)
        res2 = set(reult2)
        res3 = set(reult3)
        res4 = set(reult4)

        if len(res1) == 1 & len(res2) == 1 & len(res3) == 1 & len(res4) == 1:
            if res1 == res2 == res3 == res4:
                name = list(res1)[0]
                s = 800
                # print(list(res1)[0])
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
            te = clf.predict_proba(data1)
            # print(te.shape)
            print(np.argmax(np.bincount(res)))
            print(np.sum(res == np.argmax(np.bincount(res))))
            name = np.argmax(np.bincount(res))
            s = np.sum(res == np.argmax(np.bincount(res)))
        score.append([s])
        name_label.append([name])

        print(label[m])
        m += 1
    score = np.array(score)
    name_label = np.array(name_label)
    return score,name_label

def svm_pic():
    m = 0
    data, label = load_test()
    score = []
    name_label = []
    class_in = []  # 定义类内相似度列表
    class_each = []  # 定义类间相似度列表
    for data1 in data:
        # print(len(data1))
        scaler = joblib.load(PATH + "/Model/scaler2.m")
        data1 = scaler.transform(data1)
        data1 = data1[200:600]
        pro = 0
        clf_1 = joblib.load("./Model/svm5.m")
        clf_2 = joblib.load("./Model/svm6.m")
        clf_3 = joblib.load("./Model/svm7.m")
        clf_4 = joblib.load("./Model/svm8.m")
        reult1 = clf_1.predict(data1)
        reult2 = clf_2.predict(data1)
        reult3 = clf_3.predict(data1)
        reult4 = clf_4.predict(data1)

        res1 = set(reult1)
        res2 = set(reult2)
        res3 = set(reult3)
        res4 = set(reult4)


        if len(res1) == 1 & len(res2) == 1 & len(res3) == 1 & len(res4) == 1:
            if res1 == res2 == res3 == res4:
                name = list(res1)[0]
                s = 800
                class_in.append(1)
                print(list(res1)[0])
        else:
            train2_y = np.hstack((reult1, reult2, reult3, reult4))
            train2 = np.row_stack((data1, data1, data1, data1))
            train2_y = np.array(train2_y)
            clf = svm.SVC(kernel='rbf', C=100, gamma=0.01, probability=True)
            clf.fit(train2, train2_y)
            res = clf.predict(data1)
            te = clf.predict_proba(data1)
            # print(r.index(48))
            name = np.argmax(np.bincount(res))
            print(name)
            s = np.sum(res == np.argmax(np.bincount(res)))
            print(s)
            r = list(set(res))
            # print(r)
            # if s>200:
            for i in range(len(r)):
                if r[i] == name:
                    class_in.append(np.sum(res==r[i])/400)
                else:
                    class_each.append(np.sum(res==r[i])/400)
            # else:
            #     for i in range(len(r)):
            #         class_each.append(np.sum(res == r[i]) / 800)



            # print(res)
            # print(te.shape)
        print(label[m])
        m += 1
    # print(class_each)
    # print(class_in)
    FRR = []
    FAR = []
    thresld = np.arange(0, 0.9, 0.01)  # 生成模型阈值的等差列表
    eer = 1
    for i in range(len(thresld)):
        frr = np.sum(class_in < thresld[i]) / len(class_in)
        FRR.append(frr)

        far = np.sum(class_each > thresld[i]) / len(class_each)
        FAR.append(far)

        if (abs(frr - far) < 0.02):  # frr和far值相差很小时认为相等
                # print(frr,far)
            eer = abs(frr + far) / 2

    # print(FAR,FRR)
        # auc = metrics.auc(FAR, FRR)
        # print(auc)
    plt.plot(thresld, FRR, 'x-', label='FRR')
    plt.plot(thresld, FAR, '+-', label='FAR')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
    print('EER is: ', eer)


def pred_svm2():
    data = pre_data()
    scaler = joblib.load(PATH + "/Model/scaler2.m")
    data1 = scaler.transform(data)
    data1 = data1[0:800]
    clf_1 = joblib.load(PATH + "/Model/svm5.m")
    clf_2 = joblib.load(PATH + "/Model/svm6.m")
    clf_3 = joblib.load(PATH + "/Model/svm7.m")
    clf_4 = joblib.load(PATH + "/Model/svm8.m")
    reult1 = clf_1.predict(data1)
    reult2 = clf_2.predict(data1)

    reult3 = clf_3.predict(data1)

    reult4 = clf_4.predict(data1)

    res1 = set(reult1)
    res2 = set(reult2)
    res3 = set(reult3)
    res4 = set(reult4)

    if len(res1) == 1 & len(res2) == 1 & len(res3) == 1 & len(res4) == 1:
        if res1 == res2 == res3 == res4:
            name_label = list(res1)[0]
            score = 800
    else:
        train2_y = np.hstack((reult1, reult2, reult3, reult4))
        train2 = np.row_stack((data1, data1, data1, data1))
        train2_y = np.array(train2_y)
        clf = svm.SVC(kernel='rbf', C=100, gamma=0.01, probability=True)
        clf.fit(train2, train2_y)
        res = clf.predict(data1)
        name_label = np.argmax(np.bincount(res))
        score = np.sum(res == np.argmax(np.bincount(res)))
    return score, name_label


if __name__ == '__main__':
    # svm_2()
    test2()
    # svm_pic()
