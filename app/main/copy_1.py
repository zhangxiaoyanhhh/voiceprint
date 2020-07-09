from sklearn import svm

from app.main.feature import load_test
from app.main.model import Session, Tvoice,engine
from sklearn.model_selection import KFold
import numpy as np
from sklearn import model_selection

session = Session()
train_x = []
Y = []
j = 0
b = session.query(Tvoice).all()
for i in b:
    x = np.frombuffer(i.feature).reshape(i.qian, i.hou)

    train_x.append(x)

    for m in range(i.qian):
        Y.append(j)
    j = j+1
data = train_x[0]
for i in range(1,len(train_x)):
    print(i)
    data = np.row_stack((data ,train_x[i]))
    # print(data.shape)
Y = np.array(Y)
# kernel=['rbf','sigmoid']
# C=[8.5,8,9,9.5]
#
# parameters = {'kernel':kernel,'C':C}
# grid_svc = model_selection.GridSearchCV(estimator = svm.SVC(),
#                                         param_grid =parameters,
#                                         scoring='accuracy',cv=2,verbose =1)
# # 模型在训练数据集上的拟合
# X_train = np.array(data)
# y_train = np.array(Y)
# grid_svc.fit(X_train,y_train)
# # 返回交叉验证后的最佳参数值
# print(grid_svc.best_params_, grid_svc.best_score_)

clf1 = svm.SVC(C=9.5, decision_function_shape='ovo',probability=True)
clf1.fit(data, Y)

m = 0
data, label = load_test()
for data1 in data:
    data1 = data1[200:1000]
    res = clf1.predict(data1)
    print(np.argmax(np.bincount(res)))
    print(np.sum(res == np.argmax(np.bincount(res))))
    print(label[m])
    m += 1