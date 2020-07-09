import numpy
from scipy.io import wavfile
from matplotlib import pyplot as plt
from librosa import to_mono
import python_speech_features as psf
from librosa import feature
import librosa.display
import librosa
from functools import reduce
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
import os
from app.main.model import Tvoice, Session
from app.main.test import filter,write_wave
import scipy
import wave

session = Session()

PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))# + "/voice/"

def mfcc(path = PATH):
    # in_wav = PATH + '/voice/'+path
    # # print(path)
    # out_dir = os.path.abspath(os.path.join(in_wav,os.path.pardir))
    # # print(out_dir)
    # filter(in_wav, out_dir, expand=False)
    # (rate, signal) = wavfile.read(os.path.abspath(os.path.join(out_dir,os.path.pardir))+'/'+path) # 采样频率rate， 信号数组signal

    (rate, signal) = wavfile.read(path)#1
    signal = np.array(signal, np.float32)
    # signal = librosa.resample(signal, rate, 32000)
    # rate = 32000
    # print(rate)

    # print(len(signal))
    # f1 = plt.figure()
    # ax1 = f1.add_subplot(2, 1, 1)
    # ax1.plot(signal)
    # 预加重
    # pre_emphasis = 0.97
    # emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    #
    # signal = psf.sigproc.preemphasis(signal, coeff=0.97)

    emphasized_signal = to_mono(signal.T)

    # ax2 = f1.add_subplot(2,1,2)
    # ax2.plot(emphasized_signal)
    #
    emphasized_signal = emphasized_signal/max(abs(emphasized_signal))
    # ax2 = f1.add_subplot(2, 1, 2)
    # ax2.plot(emphasized_signal)
    # plt.show()

    mfcc_feats = psf.mfcc(signal=emphasized_signal, samplerate=rate,winlen=0.02,winstep=0.01, numcep=13,nfft=2048,nfilt=28,winfunc=lambda x:np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (x - 1)) for n in range(x)])+numpy.ones((x,)))
    # # print(mfcc_feats.mean( axis=0))
    # mfcc_feats -= (numpy.mean(mfcc_feats, axis=0))

    d_mfcc_feat = psf.delta(mfcc_feats, 1)
    d_mfcc_feat2 = psf.delta(mfcc_feats, 2)
    mfcc_feats = np.hstack((mfcc_feats, d_mfcc_feat, d_mfcc_feat2))
    # print(numpy.mean(mfcc_feats, axis=0))
    # mfcc_feats -= (numpy.mean(mfcc_feats, axis=0))


    # mfcc_feats = feature.mfcc(y=emphasized_signal, sr=rate,n_mfcc=13)
    # mfcc_feats = mfcc_feats.T
    # d_mfcc_feat = psf.delta(mfcc_feats, 1)
    # d_mfcc_feat2 = psf.delta(mfcc_feats, 2)
    # mfcc_feats = np.hstack((mfcc_feats, d_mfcc_feat, d_mfcc_feat2))
    # mfcc_feats = mfcc_feats.T
    # mfcc_feats -= (numpy.mean(mfcc_feats, axis=0) + 1e-8)
    # print(mfcc_feats.shape)

    # print(len(mfcc_feats[0]))
    # mfcc_feats = mfcc_feats.T
    # average = reduce(lambda acc, ele: acc + ele, mfcc_feats)
    # average = list(map(lambda x: x/len(mfcc_feats), average))
    # for j, feature_vector in enumerate(mfcc_feats):
    #     for k, fea in enumerate(feature_vector):
    #         mfcc_feats[j][k] -= average[k]
    # mfcc_feats = mfcc_feats.T
    # # print(mfcc_feats)
    # np.savetxt('feature.csv', mfcc_feats.T, delimiter = ',')
    # plt.show()
    # data_shape = mfcc_feats.shape
    # data_rows = data_shape[0]
    # data_cols = data_shape[1]
    #
    # data_col_max = mfcc_feats.max(axis=0)  # 获取二维数组列向最大值
    # data_col_min = mfcc_feats.min(axis=0)  # 获取二维数组列向最小值
    # for i in range(0, data_rows, 1):  # 将输入数组归一化
    #     for j in range(0, data_cols, 1):
    #         mfcc_feats[i][j] = (mfcc_feats[i][j] - data_col_min[j]) / (data_col_max[j] - data_col_min[j])
    # mfcc_feats = mfcc_feats.T
    # d_mfcc_feat = psf.delta(mfcc_feats, 1)
    # d_mfcc_feat2 = psf.delta(mfcc_feats, 2)
    # mfcc_feats = np.hstack((mfcc_feats, d_mfcc_feat, d_mfcc_feat2))
    # print(mfcc_feats)

    return mfcc_feats


def load_train(name, wav):
    # wav_path = os.listdir(PATH + '/voice/train/')
    # data = []
    # label = []
    # for wav in wav_path:
    #     if wav:
    #         label.append(wav[:-4])
    #         data.append(mfcc(PATH + '/voice/train/' + wav))# 1
    #         # data.append(mfcc('/train/'+wav))
    #         label1 = wav[:-4]
    #         a = mfcc(PATH + '/voice/train/' + wav)#1
    #         # a = mfcc('/train/'+wav)
    #         # print(len(a))
    #         file_info = Tvoice(label=label1, feature=a.tostring(), qian=len(a), hou=len(a[0]))
    #         session.add(file_info)
    #         session.commit()
    #         session.close()
    label = name
    data = mfcc(wav)

    label1 = name
    a = mfcc(wav)
    a1 = a
    file_info = Tvoice(label=label1, feature=a1.tostring(), qian=len(a1), hou=len(a1[0]))
    session.add(file_info)
    session.commit()
    session.close()
    return data, label


def pre_data():
    wav_path = PATH + '/bishe2/app/voice/test/test.wav'
    # in_wav = wav_path
    # # print(wav_path)
    # out_dir = os.path.abspath(os.path.join(in_wav,".."))
    # # print(out_dir)
    # filter(in_wav, out_dir, expand=False)
    data = mfcc(wav_path)
    return data



def load_test():
    wav_path = os.listdir(PATH + '/voice/test/')
    data = []
    label = []
    for wav in wav_path:
        if wav:
        #     in_wav = PATH + '/voice/test/' + wav
        #     # print(path)
        #     out_dir = os.path.abspath(os.path.join(in_wav,".."))
        #     # print(out_dir)
        #     filter(in_wav, out_dir, expand=False)
            # print(PATH + 'test/' + wav)
            label.append(wav[:-4])
            data.append(mfcc(PATH + '/voice/test/' + wav))#1
            # data.append(mfcc('/test/'+wav))
    # wav_path = PATH + '/bishe2/app/voice/test/test.wav'
    # data = mfcc(wav_path)
    # print(data)
    # data = [data]
    return data, label


def load_test1():
    wav_path = PATH + '/bishe2/app/voice/test/test.wav'
    # data = mfcc(wav_path)
    # print(len(data))
    # wav_path = PATH + '/voice/test/test - 副本.wav'
    # in_wav = wav_path
    # # print(path)
    # out_dir = os.path.abspath(os.path.join(in_wav,".."))
    # # print(out_dir)
    # filter(in_wav, out_dir, expand=False)
    data = mfcc(wav_path)
    print(len(data))
    data = [data]
    return data

def pic():
    # wav_path = os.listdir(PATH + '/voice/train/')
    # for wav in wav_path:
    #     in_wav = PATH + '/voice/train/'+wav
    #     print(in_wav)
    #     out_dir = PATH + '/voice/train/'
    #     # print(out_dir)
    #     filter(in_wav, out_dir, expand=False)

        filename = PATH + '/voice/train/A2.wav'
        # 打开语音文件。
        f = wave.open(filename, 'rb')
        # 得到语音参数
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # --------------------------------------------------------------#
        # 将字符串格式的数据转成int型
        print("reading wav file......")
        strData = f.readframes(nframes)
        waveData = np.fromstring(strData, dtype=np.short)
        # 归一化
        waveData = waveData * 1.0 / max(abs(waveData))
        # 将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
        waveData = np.reshape(waveData, [nframes, nchannels]).T  # .T 表示转置

        f.close()  # 关闭文件
        print("plotting spectrogram...")
        framelength = 0.025  # 帧长20~30ms
        framesize = framelength * framerate  # 每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等\
        # 而NFFT最好取2的整数次方,即framesize最好取的整数次方
        # 找到与当前framesize最接近的2的正整数次方
        nfftdict = {}
        lists = [32, 64, 128, 256, 512, 1024]
        for i in lists:
            nfftdict[i] = abs(framesize - i)
        sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
        framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize
        NFFT = framesize  # NFFT必须与时域的点数framsize相等，即不补零的FFT
        overlapSize = 1.0 / 3 * framesize  # 重叠部分采样点数overlapSize约为每帧点数的1/3~1/2
        overlapSize = int(round(overlapSize))  # 取整
        print("帧长为{},帧叠为{},傅里叶变换点数为{}".format(framesize, overlapSize, NFFT))


        spectrum, freqs, ts, fig = plt.specgram(waveData[0], NFFT=NFFT, Fs=framerate, window=np.hanning(M=framesize),
                                        noverlap=overlapSize, mode='default', scale_by_freq=True, sides='default',
                                        scale='dB', xextent=None)  # 绘制频谱图

        # print(spectrum,freqs,ts,fig)
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        plt.title("Spectrogram")
        plt.show()


    #     # print(wav)
    # y,sr = librosa.load(PATH + '/voice/train/A2.wav')
    # melspec = librosa.feature.melspectrogram(y, sr=sr, n_fft=512, n_mels=128)
    #         # melspec = melspec[:100]
    # logmelspec = librosa.power_to_db(melspec)  # 转换为对数刻度
    # plt.figure()
    #         # plt.rcParams['figure.dpi'] = 100  # 分辨
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis ='mel')
    # plt.colorbar(format='% +2.0fdB')  # 右边的色度条
    # plt.title('spectrogram')
            # plt.axis('off')
            # plt.axes().get_xaxis().set_visible(False)
            # plt.axes().get_yaxis().set_visible(False)
            # plt.cm.gray_r
            # print(PATH)
            # plt.savefig(PATH + '/voice/train_png/'+wav[:-4]+str(i),dpi=50,bbox_inches='tight', pad_inches=0)
            # plt.close('all')
            # plt.clf()
    # plt.show()


if __name__ == '__main__':
    # pic()
    # load_test1()
    data,label = load_train()
