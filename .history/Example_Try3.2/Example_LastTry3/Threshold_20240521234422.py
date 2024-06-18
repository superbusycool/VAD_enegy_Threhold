from Feature_extraction.Enframe import*
from Data_Analysis.Timefeature import *
import numpy as np
import os
import librosa


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


#关于文件读取,存档
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.wav'):
                fullname = os.path.join(root, f)
                yield fullname


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
 
        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'


# 定义要创建的目录
mkpath = "D:\\example_lastTry3.1\\Data_predict\\"
# 调用函数
mkdir(mkpath)


base='D:\example_lastTry3.1\Example_LastTry3\data_sheet'
for file_c in findAllFile(base):
    file_name= os.path.splitext(os.path.basename(file_c))
    File_name_Real=file_name[0]
    data, fs =librosa.core.load(file_c,sr=8000)
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

    full_path = mkpath + File_name_Real + '.txt'  # 创建.txt,存放预测结果
    file=open(full_path,'w')

    # print('\n')
    # print(File_name_Real)
    # print('\n')


    for i in range(vsl):
        
        Start=int(frameTime[voiceseg[i]['start']]*10000)
        End=int(frameTime[voiceseg[i]['end']]*10000)
        msg = str(Start)+','+str(End)+'\n'
        file.write(msg)
        # print(msg)
        Start=0
        End=0
    # print('\n')
    # print(File_name_Real)
    # print('\n')
    file.close()


  