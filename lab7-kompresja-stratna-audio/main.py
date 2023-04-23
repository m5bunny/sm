import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib.pyplot import figure

A = 87.6
MU = 255

def changeBitRes(src, bits):
    src_data_type = src.dtype
    if np.issubdtype(src_data_type,np.integer):
        src_m = np.iinfo(src_data_type).min
        src_n = np.iinfo(src_data_type).max
    else:
        src_m = -1
        src_n = 1
       
    src = src.astype(float)
    d = 2**bits -1
    src = (src - src_m)/(src_n-src_m)
    src = np.round(src*d)/d * (src_n-src_m) + src_m
    src = src.astype(src_data_type)

    return src

def aLawCompress(s):
    s = np.array(s)

    fcond = np.abs(s) < 1 / A
    scond = (np.abs(s) >= 1/A) & (np.abs(s) <=1)
    s[fcond] = np.sign(s[fcond]) * (A * np.abs(s[fcond]) / (1 + np.log(A)))
    s[scond] = np.sign(s[scond]) * ((1 + np.log(A * np.abs(s[scond]))) / (1 + np.log(A))) 

    return s

def aLawDecompress(s):
    s = np.array(s)

    fcond = np.abs(s) < (1 / (1 + np.log(A)))
    scond = (np.abs(s) >= (1 / (1 + np.log(A)))) & (np.abs(s) <= 1)
    s[fcond] = (np.sign(s[fcond]) * ((np.abs(s[fcond]) * (1 + np.log(A))))) / A
    s[scond] = np.sign(s[scond]) * ((np.exp(np.abs(s[scond]) * (1 + np.log(A)) - 1))) / A

    return s

def muLawCompress(s):
    s = np.array(s)

    cond = (-1 <= s) & (s <= 1)
    s[cond] = np.sign(s[cond]) * np.log(1 + MU * np.abs(s[cond])) / np.log(1 + MU);

    return s

def muLawDecompress(s):
    s = np.array(s)

    cond = (-1 <= s) & (s <= 1)
    s[cond] = np.sign(s[cond]) * 1 / MU * (np.power((1 + MU), np.abs(s[cond])) - 1)

    return s

def dpcmCompress(s, newBits):
    arr = []
    e = 0
    for i in range(len(s)):
        yP = s[i] - e
        y = changeBitRes(yP, newBits)
        arr.append(y)
        e += y
    return np.array(arr)

def dpcmDecompress(s): 
    arr = []
    tmp = 0
    for i in range(len(s)):
        xP = s[i] + tmp
        arr.append(xP)
        tmp = xP
    return np.array(arr)

def dpcmCompressPredict(s, newBits):
    arr = []
    xP = []
    e = []
    for i in range(len(s)):
        yP = s[i] - (0 if i < 1 else e[i - 1])
        y = changeBitRes(yP, newBits)
        arr.append(y)
        xP.append(y if i < 1 else y + e[i - 1])
        if (i == 0):
            e.append(xP[i]);
        elif (i == 1):
            e.append(np.mean([xP[i - 1], xP[i]]))
        else:
            e.append(np.mean([xP[i-2], xP[i-1], xP[i]]));

    return np.array(arr)

def dpcmDecompressPredict(s): 
    xP = []
    e = []
    for i in range(len(s)):
        xP.append(s[i] + (0 if i < 1 else e[i - 1]))
        if (i == 0):
            e.append(xP[i]);
        elif (i == 1):
            e.append(np.mean([xP[i - 1], xP[i]]))
        else:
            e.append(np.mean([xP[i-2], xP[i-1], xP[i]]));

    return np.array(xP)
