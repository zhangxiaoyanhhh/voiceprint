from app.main.feature import load_test1,load_test,pre_data
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl
from app.main.model import Session, Tvoice
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import matplotlib as mpl

session = Session()
path = os.path.abspath(os.path.dirname(__file__))


def train(no_components, model = False):
    # load_train()
    print("training")
    train_data = []
    b = session.query(Tvoice).all()
    for i in b:
        x = np.frombuffer(i.feature).reshape(i.qian, i.hou)
        # print(x)
        train_data.append(x)

    # print(len(train_data))
    train_model_size = len(train_data)
    TOTAL = train_model_size
    if model:
        print("load model from file...")
        with open(path+"/Model/GMM_MFCC_model.pkl", 'rb') as f:
            gmm = pkl.load(f)
        with open(path+"/Model/UBM_MFCC_model.pkl", 'rb') as f:
            ubm = pkl.load(f)
    else:
        gmm = []
        ubm_train = None
        Flag = False
        for i in range(train_model_size):
            print(i)
            gmm.append(GaussianMixture(n_components=no_components, covariance_type='tied'))
            gmm[i].fit(train_data[i])
            if Flag:
                ubm_train = np.vstack((ubm_train, train_data[i]))
            else:
                ubm_train = train_data[i]
                Flag = True
        ubm = GaussianMixture(n_components=no_components, covariance_type='tied')
        ubm.fit(ubm_train)
        # data_plot(ubm_train)

        if not os.path.exists('Model'):
            os.mkdir("Model")
        with open(path+"/Model/GMM_MFCC_model.pkl", 'wb') as f:
            pkl.dump(gmm, f)
        with open(path+"/Model/UBM_MFCC_model.pkl", 'wb') as f:
            pkl.dump(ubm, f)
    return TOTAL


def predict(TOTAL):
    print("load model from file...")
    with open(path+"/Model/GMM_MFCC_model.pkl", 'rb') as f:
        gmm = pkl.load(f)
    with open(path+"/Model/UBM_MFCC_model.pkl", 'rb') as f:
        ubm = pkl.load(f)
    # test_data = load_test1()
    test_data, test_label = load_test()
    test_model_size = len(test_data)
    # print(test_data)
    # avg_accuracy = 0
    confusion = [[0 for y in range(TOTAL)] for x in range(test_model_size)]
    # print(confusion)
    for i in range(test_model_size):
        for j in range(TOTAL):
            # print(len(test_data[i]))
            x = gmm[j].score_samples(test_data[i][:800]) - ubm.score_samples(test_data[i][:800])
            # print(x)
            for score in x:
                # print(score)
                if score > 0:
                    confusion[i][j] += 1
    # print(confusion)
    j = 0
    score1 = []
    name_label1 = []
    auc = 0
    for i in confusion:
        # print(i)
        score = max(i)
        print(score)

        name_label = i.index(max(i))
        print(name_label)
        score1.append([score])
        name_label1.append([name_label])


        # if score>600:
        #     auc += score
        #     name_label1.append(name_label)
        #     score1.append(score / 800)
        #     confusion[j].append(0)
        #     print(score)
        #     print(name_label)
        #     print(test_label[j])
        # else:
        #     name_label1.append(54)
        #     confusion[j].append(800-score)
        #     score1.append(1-(score / 800))

        print(test_label[j])
        j += 1
    score1 = np.array(score1)
    # print(np.array(confusion).shape)
    #     score1.append(score/800)
    # print((auc/33)/800)
    # print(name_label1)
    name_label1 = np.array(name_label1)
    return score1, name_label1


def data_plot(data):
    fig = plt.figure()
    plt.scatter(data[:, 1], data[:, 2])
    plt.show()


def pred_ubm(TOTAL):
    print("load model from file...")
    with open(path + "/Model/GMM_MFCC_model.pkl", 'rb') as f:
        gmm = pkl.load(f)
    with open(path + "/Model/UBM_MFCC_model.pkl", 'rb') as f:
        ubm = pkl.load(f)
    test_data = load_test1()
    test_model_size = len(test_data)
    confusion = [[0 for y in range(TOTAL)] for x in range(test_model_size)]
    for i in range(test_model_size):
        for j in range(TOTAL):
            x = gmm[j].score_samples(test_data[i][:800]) - ubm.score_samples(test_data[i][:800])
            for score in x:
                if score > 0:
                    confusion[i][j] += 1
    for i in confusion:
        score = max(i)
        name_label = i.index(max(i))
    return score,name_label


if __name__ == '__main__':
    TOTAL = train(30)
    # print(TOTAL)
    y_score,y_one_hot = predict(56)
    # print(len(y_score))
    #
    # y_score = np.array(y_score)/800
    # class_in = []  # 定义类内相似度列表
    # class_each = []  # 定义类间相似度列表
    # user_id_length = len(y_score[0])  # 要识别的数量
    # model_id_length = len(y_one_hot)  # 计算出模型ID数量
    # print(user_id_length)
    # print(model_id_length)
    # print(y_score.shape)
    #
    # for i in range(user_id_length):
    #     for j in range(model_id_length):
    #         # 需要识别的用户id和模型id一样，就认为是类内测试，否则是类间测试
    #         if i == y_one_hot[j]:
    #             class_in.append(np.float(y_score[j][i]))
    #         else:
    #             class_each.append(np.float(y_score[j][i]))
    # FRR = []
    # FAR = []
    # thresld = np.arange(0.1, 0.9, 0.01)  # 生成模型阈值的等差列表
    # eer = 1
    # for i in range(len(thresld)):
    #     frr = np.sum(class_in < thresld[i]) / len(class_in)
    #     FRR.append(frr)
    #
    #     far = np.sum(class_each > thresld[i]) / len(class_each)
    #     FAR.append(far)
    #
    #     if (abs(frr - far) < 0.02):  # frr和far值相差很小时认为相等
    #         # print(frr,far)
    #         eer = abs(frr + far) / 2
    #
    # # print(FAR,FRR)
    # # auc = metrics.auc(FAR, FRR)
    # # print(auc)
    # plt.plot(thresld, FRR, 'x-', label='FRR')
    # plt.plot(thresld, FAR, '+-', label='FAR')
    # plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    # plt.show()
    # print('EER is: ', eer)


