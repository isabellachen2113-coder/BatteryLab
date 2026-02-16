# utils/nasa_loader.py
import scipy.io
from datetime import datetime
import os
import numpy as np

def convert_to_time(hmm):
    """将MATLAB日期向量转换为datetime对象"""
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

def loadMat(matfile):
    """加载MAT文件并解析为结构化的电池数据列表"""
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]
    col = data[filename]
    col = col[0][0][0][0]  # 根据NASA数据结构解包
    size = col.shape[0]
    result = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d2 = {}
        # 跳过 impedance 类型（如果不需要）
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
            d1 = {
                'type': str(col[i][0][0]),
                'temp': int(col[i][1][0]),
                'time': convert_to_time(col[i][2][0]),
                'data': d2
            }
            result.append(d1)
    return result

def getBatteryCapacity(Battery):
    """从加载的电池数据中提取容量序列"""
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]

def getBatteryValues(Battery, Type='charge'):
    """提取指定类型（charge/discharge）的所有循环数据"""
    data = []
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data