import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.fftpack


def changeBitRes(data, resBits):
    dataType = data.dtype

    minValue = np.iinfo(dataType).min if np.issubdtype(dataType, np.integer) else -1
    maxValue = np.iinfo(dataType).max if np.issubdtype(dataType, np.integer) else 1

    d=(2**resBits)-1

    convData = (data.copy()).astype(np.float32)
    convData = (convData - minValue) / (maxValue - minValue)
    convData = np.round(convData * d)
    convData = convData / d
    convData = ((convData * (maxValue - minValue)) + minValue).astype(dataType)
    return convData 

def decimateData(data, rate):
    return data[0::rate]

def linearInterpolation(time, interpolatedTime, data):
    lf = interp1d(time, data)
    return lf(interpolatedTime).astype(data.dtype)

def nonLinearInterpolation(time, interpolatedTime, data):
    nonLf = interp1d(time, data)
    return nonLf(interpolatedTime).astype(data.dtype)

