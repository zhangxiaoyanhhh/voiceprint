from sklearn import svm
import os
from app.main.feature import load_test
from app.main.model import Session, Tvoice,engine
from sklearn.model_selection import KFold
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection

PATH = os.path.abspath(os.path.dirname(__file__))



session = Session()
def svm_one():

    train_x = []
    Y = []
    j = 1
    b = session.query(Tvoice).all()
    min_data = []
    for i in b:
        x = np.frombuffer(i.feature).reshape(i.qian, i.hou)
        a = int((i.qian)/2)
        min_data.append(a)
        x1 = x[200:600]
        x2 = x[600:1000]
        train_x.append(x1)
        train_x.append(x2)
        # for m in range(i.qian):
        Y.append(j)
        Y.append(j)
        j += 1
    n = min(min_data)

    # n = 1200
    kf = KFold(n_splits=2*(j-1))
    train_x = np.array(train_x)
    Y = np.array(Y)
    train2_y = []
    train2 = []


    for train_index, test_index in kf.split(train_x):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        data = X_train[0]
        print(train_index)
        np.set_printoptions(threshold=np.inf)
        for i in range(1,len(X_train)):
            data = np.row_stack((data ,X_train[i]))

        j = 0
        X = []
        for i in X_train:
            for m in range(len(i)):
                X.append(y_train[j])
            j += 1
        clf1 = svm.SVC(C=5, decision_function_shape='ovo',probability=True)
        clf1.fit(data, X)
        joblib.dump(clf1, PATH + "/Model/svm1.m")
        reult1 = clf1.predict_proba(X_test[0])
        for i in range(len(reult1)):
            train2_y.append(y_test[0])
        train = reult1

        clf2 = svm.SVC(C=4.5,decision_function_shape='ovo',probability=True)
        clf2.fit(data, X)
        reult2 = clf2.predict_proba(X_test[0])
        train = np.row_stack((train, reult2))
        joblib.dump(clf2, PATH + "/Model/svm2.m")
        for i in range(len(reult2)):
            train2_y.append(y_test[0])

        clf3 = svm.SVC(C=4, decision_function_shape='ovo',probability=True)
        clf3.fit(data, X)
        reult3 = clf3.predict_proba(X_test[0])
        train = np.row_stack((train, reult3))
        joblib.dump(clf3, PATH + "/Model/svm3.m")
        for i in range(len(reult3)):
            train2_y.append(y_test[0])


        clf4 = svm.SVC(C=4.7, decision_function_shape='ovo',probability=True)
        clf4.fit(data, X)
        reult4 = clf4.predict_proba(X_test[0])
        train = np.row_stack((train, reult4))
        joblib.dump(clf4, PATH + "/Model/svm4.m")
        # print(train.shape)
        for i in range(len(reult4)):
            train2_y.append(y_test[0])
        train = train.tolist()
        train2 = train2 + train
        print(len(train2))

    train2 = np.array(train2)
    train2_y = np.array(train2_y)
    print(train2.shape)
    print(train2_y.shape)
    # kernel=['rbf','linear']
    # C=[0.0001,0.001,0.05]
    #
    # parameters = {'kernel':kernel,'C':C}
    # grid_svc = model_selection.GridSearchCV(estimator = svm.SVC(),
    #                                         param_grid =parameters,
    #                                         scoring='accuracy',cv=2,verbose =1)
    # # 模型在训练数据集上的拟合
    # X_train = np.array(train2)
    # y_train = np.array(train2_y)
    # grid_svc.fit(X_train,y_train)
    # # 返回交叉验证后的最佳参数值
    # print(grid_svc.best_params_, grid_svc.best_score_)

    clf = svm.SVC(C=0.1, kernel='linear',decision_function_shape='ovo', probability=True)
    clf.fit(train2, train2_y)
    joblib.dump(clf, PATH + "/Model/svm.m")



if __name__ == '__main__':
    svm_one()
    # n = 1000
    m = 0
    data, label = load_test()
    for data1 in data:
        data1 = data1[:400]
        clf_1 = joblib.load(PATH + "/Model/svm1.m")
        clf_2 = joblib.load(PATH + "/Model/svm2.m")
        clf_3 = joblib.load(PATH + "/Model/svm3.m")
        clf_4 = joblib.load(PATH + "/Model/svm4.m")
        clf_0 = joblib.load(PATH + "/Model/svm.m")
        # pro = 0
        res1 = clf_1.predict_proba(data1)
        # res_1 = []
        # for i in res1:
        #     if max(i) < 0.8:
        #         res_1.append(-1)
        # # print(len(res_1))
        # pro = pro + len(res_1)
        res2 = clf_2.predict_proba(data1)
        # res_2 = []
        # for i in res2:
        #     if max(i) < 0.8:
        #         res_2.append(-1)
        # # print(len(res_2))
        # pro = pro + len(res_2)
        res3 = clf_3.predict_proba(data1)
        # res_3 = []
        # for i in res3:
        #     if max(i) < 0.8:
        #         res_3.append(-1)
        # # print(len(res_3))
        # pro = pro + len(res_3)
        res4 = clf_4.predict_proba(data1)
        # res_4 = []
        # for i in res4:
        #     if max(i) < 0.8:
        #         res_4.append(-1)
        # # print(len(res_4))
        # pro = pro + len(res_4)
        # print(pro)
        test = res1.tolist()+res2.tolist()+res3.tolist()+res4.tolist()
        # print(test)
        # np.set_printoptions(threshold=np.inf)

        res = clf_0.predict(test)
        res_5 = []
        for i in clf_0.predict_proba(test):
            if max(i) < 0.9:
                res_5.append(-1)
        # print(len(res_5))
        # print(np.argmax(np.bincount(res)))
        # print(np.sum(res == np.argmax(np.bincount(res)))/4)
        #
        # print("label:" + label[m])
        m += 1