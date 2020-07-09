import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable
from app.main.model import Session, Tvoice
import numpy as np
from collections import  Counter
from app.main.feature import load_test
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
import sklearn.preprocessing as preprocessing
import os

session = Session()


BATCH_SIZE = 50
INPUT_SIZE = 39
OUTPUT_SIZE = 0
LR = 0.001
EPOCH = 100
PATH = os.path.abspath(os.path.dirname(__file__))




def train2():
    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=100,
                num_layers=2,
                batch_first=True
            )
            self.out = nn.Linear(100, OUTPUT_SIZE)

        def forward(self, x):
            r_out, (h_c, h_n) = self.rnn(x, None)
            output = self.out(r_out[:, -1, :])
            return output
    torch.manual_seed(1)
    j = 0
    dim = []
    label = []
    b = session.query(Tvoice).all()
    for i in b:
        dim.append(i.qian)
        label.append(i.label)
        j += 1
        if j == 1:
            train_x = torch.from_numpy(np.array([np.frombuffer(i.feature).reshape(i.qian, i.hou)[200:1000]])).type(
                torch.FloatTensor)
            train_y = torch.zeros(1).long()
            print(train_x.shape)

        else:
            x = torch.from_numpy(np.array([np.frombuffer(i.feature).reshape(i.qian, i.hou)[200:1000]])).type(torch.FloatTensor)
            y = (j - 1) * (torch.ones(1)).long()
            train_y = torch.cat((train_y, y), ).type(torch.LongTensor)
            train_x = torch.cat((train_x, x), 0).type(torch.FloatTensor)

    # print(type(train_x))
    # print(train_y.shape)
    # print(j)
    # scaler1 = preprocessing.StandardScaler()  # 标准化转换
    # scaler1.fit(train_x)  # 训练标准化对象
    # joblib.dump(scaler1, PATH + "/Model/scaler1.m")
    # train_x = torch.from_numpy(scaler1.transform(train_x))  # 转换数据集

    # print(train_x)
    OUTPUT_SIZE = j
    # print(x.shape)
    # x = torch.from_numpy(x).type(torch.FloatTensor)
    # print(x.shape)

    torch_dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    lstm = RNN()

    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = []
    num_correct = 0
    train_losses = []
    j=0

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(loader):
            # print(b_x.shape)
            b_x = torch.tensor(b_x, dtype=torch.float32)
            b_x = Variable(b_x.view(-1, 800, 39))
            b_y = Variable(b_y)
            pred = lstm(b_x)
            loss = loss_func(pred, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_loss = float(loss.item())
            # train_losses.append(train_loss/len(loader))
            # pred = pred.argmax(dim=1)
            # num_correct += torch.eq(pred, b_y).sum().float().item()
            # train_acc.append(num_correct/len(loader.dataset))
            # j+=1

    torch.save(lstm.state_dict(), PATH+'/Model/model.pkl')
    return train_losses,train_acc,j

def test():
    b = session.query(Tvoice).all()
    out = 0
    for i in b:
        out += 1
    class RNN(nn.Module):
        def __init__(self):
            super(RNN, self).__init__()
            self.rnn = nn.LSTM(
                input_size=39,
                hidden_size=100,
                num_layers=2,
                batch_first=True
            )
            self.out = nn.Linear(100, out)

        def forward(self, x):
            r_out, (h_c, h_n) = self.rnn(x, None)
            output = self.out(r_out[:, -1, :])
            return output

    lstm = RNN()

    lstm.load_state_dict(torch.load(PATH+'/Model/model.pkl'))


    # test_x = load_test()
    # test = test_x[200:1000]
    # pred_x = torch.unsqueeze(torch.from_numpy(test), dim=1).type(torch.FloatTensor)
    # test_out = lstm(pred_x)
    # pred_y = torch.max(test_out, 1)[1].data.squeeze()
    # pred_y = pred_y.data.numpy()
    # name_label = np.argmax(np.bincount(pred_y))
    # score = np.sum(pred_y == np.argmax(np.bincount(pred_y)))
    d = 0
    score = []
    name_label = []
    test_x,test_y = load_test()
    for test in test_x:
        # print(len(test))
        test = np.array([test[200:1000]])
        # scaler = joblib.load(PATH + "/Model/scaler1.m")
        # test = scaler.transform(test)
        pred_x = torch.unsqueeze(torch.from_numpy(test[0]), dim=1).type(torch.FloatTensor)
        # print(pred_x.shape)
        test_out = lstm(pred_x)
        pred_y = torch.max(test_out, 1)[1].data.squeeze()
        pred_y = pred_y.data.numpy()
        # print(pred_y)
        # Counter(pred_y)
        print(np.argmax(np.bincount(pred_y)))
        name_label.append(np.argmax(np.bincount(pred_y)))
        print(np.sum(pred_y == np.argmax(np.bincount(pred_y))))
        score.append(np.sum(pred_y == np.argmax(np.bincount(pred_y))))

        print(test_y[d])
        # print(label[prediction])
        d += 1
    return score, name_label

if __name__ == '__main__':
    # y2,y1,j = train2()
    test()
