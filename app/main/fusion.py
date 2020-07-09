from app.main.LSTM import test
from app.main.svm import test3
from app.main.svm2 import test2
from app.main.ubm import predict
import numpy as np
import matplotlib.pyplot as plt


def voting():
    score1,name1 = predict(54)
    score2,name2 = test()
    score3,name3 = test3()
    score4,name4 = test2()

    score = np.hstack((score1,score2,score3,score4))
    name = np.hstack((name1,name2,name3,name4))
    print(score,name)
    auc = 0
    num = 0
    class_in = []
    class_each = []
    for i in range(len(score)):
        # print(i)
        name_label = np.argmax(np.bincount(name[i]))
        x = np.argwhere(name[i]==name_label)
        # print(x)
        max_score = max(score[i][x[0][0]:(x[-1][0]+1)])
        if max_score >=500:
            # print(num)
            print(max_score/800)
            auc += (max_score/800)
            num += 1
            class_in.append(max_score/ 800)
            class_each.append(1-(max_score / 800))
        else:
            class_each.append(max_score / 800)
        print(name_label,max_score/800)
    # print(num)
    print(auc/num)

    FRR = []
    FAR = []
    thresld = np.arange(0, 1, 0.008)  # 生成模型阈值的等差列表
    eer = 1
    for i in range(len(thresld)):
        frr = np.sum(class_in < thresld[i]) / len(class_in)
        FRR.append(frr)

        far = np.sum(class_each > thresld[i]) / len(class_each)
        FAR.append(far)

        if (abs(frr - far) < 0.05):  # frr和far值相差很小时认为相等
            print(frr, far)
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


if __name__ == '__main__':
    voting()