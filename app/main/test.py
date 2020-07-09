# -*- coding: utf-8 -*-
import collections
import contextlib
import sys
import os
import wave

import webrtcvad

AGGRESSIVENESS = 3
# sample_width = 0

def read_wave(path):
    """Reads wave file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        # print(type(wf))
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        # print(sample_width)
        assert sample_width == 2
        sample_rate = wf.getframerate()
        # print(sample_rate)
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate#, sample_width


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Args:
        frame_duration_ms: The desired frame duration in milliseconds.
        audio: The PCM data.
        sample_rate: The sample rate
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, vad, frames):
    """Filters out non-voiced audio frames.

    Args:
        sample_rate: The audio sample rate, in Hz.
        vad: An instance of webrtcvad.Vad.
        frames: A source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """

    voiced_frames = []
    for idx, frame in enumerate(frames):
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if is_speech:
            voiced_frames.append(frame)

    return b''.join([f.bytes for f in voiced_frames])


def voiced_frames_expand(voiced_frames, duration=2):
    total = duration * 8000 * 2
    expanded_voiced_frames = voiced_frames
    while len(expanded_voiced_frames) < total:
        expand_num = total - len(expanded_voiced_frames)
        expanded_voiced_frames += voiced_frames[:expand_num]

    return expanded_voiced_frames


def filter(wavpath, out_dir, expand=False):
    '''Apply vad with wave file.

    Args:
        wavpath: The input wave file.
        out_dir: The directory that contains the voiced audio.
        expand: Expand the frames or not, default False.
    '''
    # print("wavpath:", wavpath)
    audio, sample_rate = read_wave(wavpath)
    # print('sample rate:%d'%sample_rate)
    vad = webrtcvad.Vad(AGGRESSIVENESS)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    voiced_frames = vad_collector(sample_rate, vad, frames)
    voiced_frames = voiced_frames_expand(voiced_frames, 2) if expand else voiced_frames
    wav_name = wavpath.split('/')[-1]
    save_path = out_dir + '/' + wav_name
    write_wave(save_path, voiced_frames, sample_rate)


def main():
    PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
    in_wav = PATH + '/voice/test/F183_test_1.wav'
    out_dir = PATH + '/voice'
    filter(in_wav, out_dir, expand=False)


if __name__ == '__main__':
    main()

# import numpy as np
# import pylab as plt
#
# fs = 8000
# fl = 0
# fh = fs/2
# bl = 1125*np.log(1+fl/700) # 把 Hz 变成 Mel
# bh = 1125*np.log(1+fh/700)
# p = 10
# NFFT=2048
# B = bh-bl
# y = np.linspace(0,B,p+2)# 将梅尔刻度等间隔
# #print(y)
# Fb = 700*(np.exp(y/1125)-1)# 把 Mel 变成 Hz
# #print(Fb)
# W2 = int(NFFT / 2 + 1)
# df = fs/NFFT
# freq = []#采样频率值
# for n in range(0,W2):
#     freqs = int(n*df)
#     freq.append(freqs)
# bank = np.zeros((p,W2))
# for k in range(1,p+1):
#     f1 = Fb[k-1]
#     f2 = Fb[k+1]
#     f0 = Fb[k]
#     n1=np.floor(f1/df)
#     n2=np.floor(f2/df)
#     n0=np.floor(f0/df)
#     for i in range(1,W2):
#         if i>=n1 and i<=n0:
#             bank[k-1,i]=(i-n1)/(n0-n1)
#         elif i>n0 and i<=n2:
#             bank[k-1,i]=(n2-i)/(n2-n0)
#     # print(k)
#     # print(bank[k-1,:])
#     plt.plot(freq,bank[k-1,:],'r')
#
# plt.ylabel('Frequency response')
# plt.xlabel('Frequency')
# plt.title("filter bank")
# plt.show()
