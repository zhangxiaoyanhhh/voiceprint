from sklearn import svm

from app.main.feature import load_test,pre_data
from app.main.model import Session, Tvoice,engine
from sklearn.model_selection import KFold
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import BaggingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
import os
import matplotlib.pyplot as plt

PATH = os.path.abspath(os.path.dirname(__file__))

session = Session()
def svm_two():

    train_x = []
    train_y = []
    Y = []
    j = 0
    b = session.query(Tvoice).all()
    for i in b:
        x = np.frombuffer(i.feature).reshape(i.qian, i.hou)
        train_x.append(x[200:1000])
        train_y.append(i.label)
        train_y.append(i.label)
        for m in range(800):
            Y.append(j)
        j += 1
    data = train_x[0]
    np.set_printoptions(threshold=np.inf)
    for i in range(1, len(train_x)):
        data = np.row_stack((data, train_x[i]))
    print(data.shape)
    Y = np.array(Y)
    print(Y.shape)
    scaler = preprocessing.StandardScaler()  # 标准化转换
    scaler.fit(data)  # 训练标准化对象
    joblib.dump(scaler, PATH + "/Model/scaler.m")
    traffic_feature= scaler.transform(data)   # 转换数据集
    traffic_target = Y
    feature_train = traffic_feature
    target_train = traffic_target
    # feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.2,random_state=0)
    # print(len(target_test))
    clf1 = svm.SVC(C=5, decision_function_shape='ovo', probability=True)
    clf1.fit(feature_train, target_train)
    joblib.dump(clf1, PATH + "/Model/svm_1.m")
    # predict_results1 = clf1.predict(feature_test)
    # print(accuracy_score(predict_results1, target_test))

    clf2 = svm.SVC(C=4.5, decision_function_shape='ovo', probability=True)
    clf2.fit(feature_train, target_train)
    joblib.dump(clf2, PATH + "/Model/svm_2.m")
    # predict_results2 = clf2.predict(feature_test)
    # print(accuracy_score(predict_results2, target_test))

    clf3 = svm.SVC(C=7, decision_function_shape='ovo', probability=True)
    clf3.fit(feature_train, target_train)
    joblib.dump(clf3, PATH + "/Model/svm_3.m")
    # predict_results3 = clf3.predict(feature_test)
    # print(accuracy_score(predict_results3, target_test))

    clf4 = svm.SVC(C=3, decision_function_shape='ovo', probability=True)
    clf4.fit(feature_train, target_train)
    joblib.dump(clf4, PATH + "/Model/svm_4.m")
    # predict_results4 = clf4.predict(feature_test)
    # print(accuracy_score(predict_results4, target_test))


    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3,clf4],
                              use_probas=True,
                              meta_classifier=svm.SVC(C=4.5, decision_function_shape='ovo', probability=True))
    sclf.fit(feature_train, target_train)
    joblib.dump(sclf, PATH + "/Model/svm0.m")
    # predict_results = sclf.predict(feature_test)
    # # print(predict_results)
    # print(accuracy_score(predict_results, target_test))

def test3():
    data, label = load_test()
    m = 0
    score = []
    name_label = []
    for data1 in data:
        data1 = data1[200:1000]
        scaler = joblib.load(PATH + "/Model/scaler.m")
        data1 = scaler.transform(data1)
        # clf1 = joblib.load(PATH + "/Model/svm_1.m")
        # clf2 = joblib.load(PATH + "/Model/svm_2.m")
        # clf3 = joblib.load(PATH + "/Model/svm_3.m")
        # clf4 = joblib.load(PATH + "/Model/svm_4.m")

        clf = joblib.load(PATH + "/Model/svm0.m")
        # res1 = clf1.predict(data1)
        # res2 = clf2.predict(data1)
        # res3 = clf3.predict(data1)
        # res4 = clf4.predict(data1)
        # c1 = np.sum(res1 == np.argmax(np.bincount(res1))) / len(data1)
        # c2 = np.sum(res2 == np.argmax(np.bincount(res2))) / len(data1)
        # c3 = np.sum(res3 == np.argmax(np.bincount(res3))) / len(data1)
        # c4 = np.sum(res4 == np.argmax(np.bincount(res4))) / len(data1)
        predict_results = clf.predict(data1)
        predict_results2 = clf.predict_proba(data1)
        b = np.argmax(np.bincount(predict_results))
        c = np.sum(predict_results == np.argmax(np.bincount(predict_results)))
        print(b)
        y = np.ones(len(data1)) * b
        print(c)
        score.append([c])
        name_label.append([b])

        # print(accuracy_score(predict_results, y))
        # if c <0.7:
        print(label[m])
        # print(a,b,c,c1,c2,c3,c4)
        m += 1
    score = np.array(score)
    name_label = np.array(name_label)
    return score,name_label


def svm2_pic():
    m = 0
    data, label = load_test()
    score = []
    name_label = []
    class_in = []  # 定义类内相似度列表
    class_each = []  # 定义类间相似度列表
    for data1 in data:
        data1 = data1[:800]
        scaler = joblib.load(PATH + "/Model/scaler.m")
        data1 = scaler.transform(data1)

        clf = joblib.load(PATH + "/Model/svm0.m")
        predict_results = clf.predict(data1)
        predict_results2 = clf.predict_proba(data1)
        b = np.argmax(np.bincount(predict_results))
        c = np.sum(predict_results == np.argmax(np.bincount(predict_results)))
        print(b)
        y = np.ones(len(data1)) * b
        print(c)
        r = list(set(predict_results))
        for i in range(len(r)):
            if r[i] == b:
                class_in.append(np.sum(predict_results==r[i])/800)
            else:
                class_each.append(np.sum(predict_results==r[i])/800)
        print(label[m])
        m += 1
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


def pred_svm1():
    data = pre_data()
    data1 = data[0:800]
    scaler = joblib.load(PATH + "/Model/scaler.m")
    data1 = scaler.transform(data1)
    clf = joblib.load(PATH + "/Model/svm0.m")
    predict_results = clf.predict(data1)
    name_label = np.argmax(np.bincount(predict_results))
    score = np.sum(predict_results == np.argmax(np.bincount(predict_results)))
    return score,name_label



if __name__ == '__main__':
    # svm_two()
    test3()
    # svm2_pic()


