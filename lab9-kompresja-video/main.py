import cv2
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

use_RLEs = [True]
clips = ['clip_1.mp4']

kat = '.'  # katalog z plikami wideo
plik = "clip.mp4"  # nazwa pliku
ile = 50  # ile klatek odtworzyc? <0 - calosc
key_frame_counter = 8  # co ktora klatka ma byc kluczowa i nie podlegac kompresji
plot_frames = np.array([35])  # automatycznie wyrysuj wykresy
auto_pause_frames = np.array([100])  # automatycznie zapauzuj dla klatki i wywietl wykres
subsampling = "4:1:1"  # parametry dla chorma subsamplingu
dzielnik = 4  # dzielnik przy zapisie różnicy
wyswietlaj_kaltki = False# czy program ma wyswietlac kolejene klatki
use_RLE = False 


##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def init(self):
        self.Y = None
        self.Cb = None
        self.Cr = None

def Chroma_subsampling(L, subsampling):
    if subsampling == '4:4:4':
        return L
    elif subsampling == '4:4:0':
        return L[::2, :]
    elif subsampling == '4:2:0':
        return L[::2, ::2]
    elif subsampling == '4:2:2':
        return L[:, ::2]
    elif subsampling == '4:1:1':
        return L[:, ::4]
    elif subsampling == '4:1:0':
        return L[::2, ::4]

def Chroma_resampling(L, subsampling):
    if subsampling == '4:4:4':
        return L
    elif subsampling == '4:4:0':
        return np.repeat(L, 2, axis=0)
    elif subsampling == '4:2:0':
        L = np.repeat(L, 2, axis=1)
        return np.repeat(L, 2, axis=0)
    elif subsampling == '4:2:2':
        return np.repeat(L, 2, axis=1)
    elif subsampling == '4:1:1':
        return np.repeat(L, 4, axis=1)
    elif subsampling == '4:1:0':
        L = np.repeat(L, 4, axis=1)
        return np.repeat(L, 2, axis=0)

def frame_image_to_class(frame):
    Frame_class = data()
    Frame_class.Y = frame[:, :, 0]
    Frame_class.Cr = frame[:, :, 1]
    Frame_class.Cb = frame[:, :, 2]
    return Frame_class

def frame_image_to_image(Frame_class):
    return np.dstack([Frame_class.Y, Frame_class.Cr, Frame_class.Cb]).clip(0, 255).astype(np.uint8)

def compress_KeyFrame(Frame_class):
    KeyFrame = data()
    KeyFrame.Y = Frame_class.Y
    KeyFrame.Cr = Chroma_subsampling(Frame_class.Cr, subsampling)
    KeyFrame.Cb = Chroma_subsampling(Frame_class.Cb, subsampling)
    return KeyFrame

def decompress_KeyFrame(Frame_class):
    if use_RLE:
        Frame_class = rleFrameDecode(Frame_class)
    Frame_class.Cr = Chroma_resampling(Frame_class.Cr, subsampling)
    Frame_class.Cb = Chroma_resampling(Frame_class.Cb, subsampling)
    # print('decompress_keyframe values: ', np.min(Frame_class.Cr), np.max(Frame_class.Cr))
    return frame_image_to_image(Frame_class)

def compress_not_KeyFrame(Frame_class, KeyFrame):
    Compress_data = data()
    Compress_data.Y = ((Frame_class.Y - KeyFrame.Y) / dzielnik).astype(np.int8)
    Compress_data.Cr = (Frame_class.Cr - KeyFrame.Cr) / dzielnik
    Compress_data.Cr = Chroma_subsampling(Compress_data.Cr, subsampling).astype(np.int8)
    Compress_data.Cb = (Frame_class.Cb - KeyFrame.Cb) / dzielnik
    Compress_data.Cb = Chroma_subsampling(Compress_data.Cb, subsampling).astype(np.int8)
    return Compress_data

def decompress_not_KeyFrame(Compress_data, KeyFrame):
    if use_RLE:
        Compress_data = rleFrameDecode(Compress_data)
    Compress_data.Cr = Chroma_resampling(Compress_data.Cr, subsampling)
    Compress_data.Cb = Chroma_resampling(Compress_data.Cb, subsampling)
    Compress_data.Y = (Compress_data.Y * dzielnik + KeyFrame.Y).astype(np.uint8)
    Compress_data.Cr = (Compress_data.Cr * dzielnik + KeyFrame.Cr).astype(np.uint8)
    Compress_data.Cb = (Compress_data.Cb * dzielnik + KeyFrame.Cb).astype(np.uint8)
    return frame_image_to_image(Compress_data)

def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI):
    # bardzo słaby i sztuczny przyklad wykrozystania tej opcji
    # przerobić żeby porównaie było dokonywane w RGB nie YCrCb i/lub zastąpić innym porównaniem
    # ROI - Region of Inrest wspłrzędne fragmentu który chcemy przybliżyć i ocenić w formacie [w1,w2,k1,k2]
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(16, 5)

    frame1 = ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    frame2 = DecompressedFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    frame1 = cv2.cvtColor(np.clip(frame1, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    axs[0].imshow(frame1)
    plt.title("File:{}, subsampling={}, divider={}, KeyFrame={} ".format(plik, subsampling, dzielnik, key_frame_counter))
    frame2 = cv2.cvtColor(np.clip(frame2, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    axs[2].imshow(frame2)
    diff = frame1.astype(float) - frame2.astype(float)
    axs[1].imshow(diff, vmin=np.min(diff), vmax=np.max(diff))




    # axs[0].imshow(ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]])
    # axs[2].imshow(DecompressedFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]])
    diff = ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(float) - DecompressedFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(float)
    # print(np.min(diff), np.max(diff))
    # axs[1].imshow(diff, vmin=np.min(diff), vmax=np.max(diff))

def rleEncrypt(dataArray):
    arrInfo = np.concatenate(([dataArray.ndim], dataArray.shape))
    flattenedArray = dataArray.ravel()

    encryptedValues = []
    count = 1

    for i in range(1, flattenedArray.size):
        if flattenedArray[i] == flattenedArray[i - 1] and count < 255:
            count += 1
        else:
            encryptedValues.extend([count, flattenedArray[i - 1]])
            count = 1

    encryptedValues.extend([count, flattenedArray[-1]])
    encrypted_array = np.concatenate((arrInfo, np.array(encryptedValues, dtype=int)))
    return encrypted_array

def rleDecrypt(encodedData):
    shape = encodedData[1:int(encodedData[0] + 1)].astype(int)
    data = encodedData[int(encodedData[0] + 1):]
    # print('data: ',data)

    size = np.prod(shape).astype(int)
    decodedData = np.empty(size, dtype=int)

    i = 0
    for j in range(0, len(data), 2):
        count = int(data[j])
        value = data[j + 1]
        decodedData[i:(i + count)] = value
        i += count
    # print('Dedoded values: ',np.min(decodedData.reshape(shape).astype(int)) ,np.max(decodedData.reshape(shape).astype(int)))
    # print('Decoded DTYPE: ',decodedData.reshape(shape).astype(int).dtype)
    return decodedData.reshape(shape).astype(int)

def rleFrameEncode(Frame_class):
    Frame_class.Y = rleEncrypt(Frame_class.Y)
    Frame_class.Cr = rleEncrypt(Frame_class.Cr)
    Frame_class.Cb = rleEncrypt(Frame_class.Cb)
    return Frame_class

def rleFrameDecode(Frame_class):
    Frame_class.Y = rleDecrypt(Frame_class.Y)
    Frame_class.Cr = rleDecrypt(Frame_class.Cr)
    Frame_class.Cb = rleDecrypt(Frame_class.Cb)
    return Frame_class

##############################################################################
####     Głowna petla programu      ##########################################
##############################################################################

for a in range(len(clips)):
  plik = clips[a]
  for b in range(len(use_RLEs)):
    use_RLE = use_RLEs[b]
    cap = cv2.VideoCapture(kat + '/' + plik)

    if ile < 0:
        ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow('Normal Frame')
    cv2.namedWindow('Decompressed Frame')

    compression_information = np.zeros((ile, 3))


    KeyFrame = None

    for i in range(ile):
        ret, frame = cap.read()
        if wyswietlaj_kaltki:
            cv2.imshow('Normal Frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Frame_class = frame_image_to_class(frame)

        if (i % key_frame_counter) == 0:
            print("Kluczowa")
            KeyFrame = compress_KeyFrame(Frame_class)
            if use_RLE:
                KeyFrameCopy = rleFrameEncode(KeyFrame)
            else:
                KeyFrameCopy = KeyFrame
            cY = KeyFrameCopy.Y
            cCr = KeyFrameCopy.Cr
            cCb = KeyFrameCopy.Cb
            Decompresed_Frame = decompress_KeyFrame(KeyFrameCopy)

        else:
            print("Niekluczowa")
            Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame)
            if use_RLE:
                CompressDataCopy = rleFrameEncode(Compress_data)
            else:
                CompressDataCopy = Compress_data
            cY = CompressDataCopy.Y
            cCr = CompressDataCopy.Cr
            cCb = CompressDataCopy.Cb
            Decompresed_Frame = decompress_not_KeyFrame(CompressDataCopy, KeyFrame)

        compression_information[i, 0] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
        compression_information[i, 1] = (frame[:, :, 1].size - cCb.size) / frame[:, :, 1].size
        compression_information[i, 2] = (frame[:, :, 2].size - cCr.size) / frame[:, :, 2].size
        print('original y size: ', frame[:, :, 0].size)
        print('compressed y size: ', cY.size)
        print('original cb size: ', frame[:, :, 1].size)
        print('compressed cb size: ', cCb.size)
        print('original cr size: ', frame[:, :, 2].size)
        print('compressed cr size: ', cCr.size)

        if wyswietlaj_kaltki:
            cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

        if np.any(plot_frames == i+1):  # rysuj wykresy
            plotDiffrence(frame, Decompresed_Frame, [250, 350, 250, 350])

        if np.any(auto_pause_frames == i):
            cv2.waitKey(-1)  # wait until any key is pressed

        k = cv2.waitKey(1) & 0xff

        if k == ord('q'):
            break
        elif k == ord('p'):
            cv2.waitKey(-1)  # wait until any key is pressed

    plt.figure()
    plt.plot(np.arange(0, ile), compression_information[:, 0] * 100)
    plt.plot(np.arange(0, ile), compression_information[:, 1] * 100)
    plt.plot(np.arange(0, ile), compression_information[:, 2] * 100)
    plt.title("File:{}, subsampling={}, divider={}, KeyFrame={}, RLE={}".format(plik, subsampling, dzielnik, key_frame_counter, use_RLE))
    plt.savefig("File:{}-subsampling={}-divider={}-KeyFrame={}-RLE={}.png".format(plik, subsampling, dzielnik, key_frame_counter, use_RLE))
    #plt.show()