from Feature_extraction.Enframe import*
from Data_Analysis.Timefeature import *
import matplotlib.pyplot as plt
import numpy as np
import audioread
import librosa
from pathlib import Path

p=Path('D:\Example_LastTry2\data_sheet')

def findSegment(express):
    """
    分割成語音段
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1]+1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1]+1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg


def vad_TwoThr(x, wlen, inc, NIS):
    """
    使用门限法检测语音段
    :param x: 语音信号
    :param wlen: 分帧长度
    :param inc: 帧移
    :param NIS:
    :return:
    """
    maxsilence = 15
    minlen = 5
    status = 0
    y = enframe(x, wlen, inc)
    fn = y.shape[0]
    amp = STEn(x, wlen, inc)
    zcr = STZcr(x, wlen, inc, delta=0.01)
    ampth = np.mean(amp[:NIS])
    zcrth = np.mean(zcr[:NIS])
    amp2 = 2 * ampth
    amp1 = 4 * ampth
    zcr2 = 2 * zcrth
    xn = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(fn):
        if status == 0 or status == 1:
            if amp[n] > amp1:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif amp[n] > amp2 or zcr[n] > zcr2:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0

        elif status == 2:
            if amp[n] > amp2 and zcr[n] > zcr2:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        elif status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF, amp, zcr


file=p.data_1.wav
data, fs =librosa.core.load(file)
data /= np.max(data)
N = len(data)
wlen = 200
inc = 80
IS = 0.1
overlap = wlen - inc
NIS = int((IS * fs - wlen) // inc + 1)
fn = (N - wlen) // inc + 1

frameTime = FrameTimeC(fn, wlen, inc, fs)
time = [i / fs for i in range(N)]

voiceseg, vsl, SF, NF, amp, zcr = vad_TwoThr(data, wlen, inc, NIS)

plt.subplot(3, 1, 1)
plt.plot(time, data)

plt.subplot(3, 1, 2)
plt.plot(frameTime, amp)

plt.subplot(3, 1, 3)
plt.plot(frameTime, zcr)

for i in range(vsl):
    plt.subplot(3, 1, 1)
    plt.plot(frameTime[voiceseg[i]['start']], 1, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 1, 'or')
    plt.show()

    plt.subplot(3, 1, 2)
    plt.plot(frameTime[voiceseg[i]['start']], 1, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 1, 'or')
    plt.show()

    plt.subplot(3, 1, 3)
    plt.plot(frameTime[voiceseg[i]['start']], 1, '.k')
    plt.plot(frameTime[voiceseg[i]['end']], 1, 'or')
    plt.show()



# plt.savefig('images/TwoThr.png')
plt.close()