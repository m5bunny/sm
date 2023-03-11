import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.fftpack
import soundfile as sf

def zad1():
    data, fs = sf.read('sound1.wav', dtype='float32')
    leftChannelData = data.copy();
    leftChannelData[:,1] = 0;
    sf.write('sound_L.wav', leftChannelData, fs)

    rigthChannelData = data.copy();
    rigthChannelData[:,0] = 0;
    sf.write('sound_R.wav', rigthChannelData, fs)

    mixedData = (data[:, 0] + data[:, 1]) / 2
    sf.write('sound_mix.wav', mixedData, fs)

    time = np.arange(0,data.shape[0])/fs
    plt.subplot(2,1,1)
    plt.plot(time, data[:,0])
    plt.subplot(2,1,2)
    plt.plot(time, data[:,1])
    plt.show()

def zad2():
    data, fs = sf.read('sin_440Hz.wav', dtype='float32')

    fsize=2**8

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.arange(0,data.shape[0])/fs,data)
    plt.subplot(2,1,2)
    yf = scipy.fftpack.fft(data,fsize)
    plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
    plt.show()

import warnings

warnings.filterwarnings('error')


def zad3():
    def plotAudio(data,fs,timeMargin=[0,0.02]):
        fsize=2**8

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.arange(0,data.shape[0])/fs,data)
        plt.xlabel("time (s)")
        plt.ylabel("dB")
        plt.xlim(timeMargin[0], timeMargin[1])

        subplot2 = plt.subplot(2,1,2)
        yf = scipy.fftpack.fft(data,fsize)
        try:
            plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
            warnings.warn(Warning())
        except:
            subplot2.clear()
            plt.plot(np.arange(0,fs/2,fs/fsize),np.abs(yf[:fsize//2]))

        plt.ylabel("dB")
        plt.xlabel("frequency (Hz)")

        plt.show()

    data, fs = sf.read('sin_440Hz.wav', dtype='float32')
    plotAudio(data, fs)
