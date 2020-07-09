import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
import os
import skimage.data
import skimage.io
import skimage.transform
import torchvision.transforms as transforms
from app.main.model import Tvoice1, Session

session = Session()

transform = transforms.ToTensor()

PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
def get_picture(path):
    img = skimage.io.imread(path)
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)
    # img248 = skimage.transform.resize(img, (248, 184))
    # print(img248.shape)
    img248 = np.asarray(img)
    img248 = img248.astype(np.float32)
    # print(img248.shape)
    return transform(img248)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,   # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(1,1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(1, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(1,1)
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(10560, 128),  # 进行线性变换
        #     nn.ReLU()  # 进行ReLu激活
        # )
        #
        # # 输出层(将全连接层的一维输出进行处理)
        # self.fc2 = nn.Sequential(
        #     nn.Linear(128, 84),
        #     nn.ReLU()
        # )

        self.out = nn.Linear(124, 124)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        output = self.out(x)
        return output


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        # print(self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                # print(name)
                x = x.view(x.size(0), -1)
            # print(module)
            x = module(x)
            # print(name)
            if name in self.extracted_layers:
                outputs.append(x.data.numpy())
        return outputs


def get_feature():
    # 输入数据
    wav_path = os.listdir(PATH + '/voice/train_png/')
    for png in wav_path:
        label1 = png[:-4]
        img = get_picture(PATH + '/voice/train_png/'+png)
        # print(img.shape)
        # 插入维度
        img = img.unsqueeze(0)

        # 特征输出
        net = CNN()
        # net.load_state_dict(torch.load('./model/net_050.pth'))
        exact_list = ["conv4"]
        myexactor = FeatureExtractor(net, exact_list)
        x = myexactor(img)
        x = np.array(x)
        x = x[0][0]
        x = x.transpose(1,2,0)#.reshape(124,2944)
        print(x.shape)
        file_info = Tvoice1(label=label1, feature=x.tostring(), one=x.shape[0], two=x.shape[1],three=x.shape[2])
        session.add(file_info)
        session.commit()
        session.close()

        # 特征输出可视化
        # for i in range(32):
        #     ax = plt.subplot(8, 8, i + 1)
        #     ax.set_title('Feature {}'.format(i))
        #     ax.axis('off')
        #     plt.imshow(x[-1][0,i,:32,:],cmap='jet')
        #
        # plt.show()

def test_pic():
    wav_path = os.listdir(PATH + '/voice/test_png/')
    data = []
    label = []
    for png in wav_path:
        label1 = png[:-4]
        label.append(label1)
        img = get_picture(PATH + '/voice/test_png/' + png)
        print(img.shape)
        # 插入维度
        img = img.unsqueeze(0)

        # 特征输出
        net = CNN()
        # net.load_state_dict(torch.load('./model/net_050.pth'))
        exact_list = ["conv4"]
        myexactor = FeatureExtractor(net, exact_list)
        x = myexactor(img)
        x = np.array(x,np.float32)
        x = x[0][0]
        x = x.transpose(1,2,0)
        # print(x)
        data.append(x)
    return label, data

# 训练
if __name__ == "__main__":
    get_feature()
    # test_pic()