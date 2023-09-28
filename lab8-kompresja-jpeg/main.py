import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2

class JPEG:
    Y=np.array([])
    Cb=np.array([])
    Cr=np.array([])
    ChromaRatio="4:4:4"
    QY=np.ones((8,8))
    QC=np.ones((8,8))
    shape=(0,0,3)

def rgbToYCrCb(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(int)

def yCrCbToRgb(yCrCb):
    return cv2.cvtColor(np.clip(yCrCb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

def dct2(block):
    return scipy.fftpack.dct(scipy.fftpack.dct(block.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho')

def idct2(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def chromaResampling(layer, type):
    if (type == "4:4:4"):
        return layer 
    if (type == "4:2:2"):
        return layer[1::2]

def chromaResamplingBack(layer, type, imgShape):
    if (type == "4:4:4"):
        return layer 
    if (type == "4:2:2"):
        biggerLayer = np.zeros(imgShape)
        layer = layer.reshape(360, 1280)
        biggerLayer[ 0::2, :: ] = layer[:,:]
        biggerLayer[ 1::2, :: ] = biggerLayer[ 0::2, :: ]
        return biggerLayer

def comressRLE(img):
    compresedImg = [];
    flatedImg = img.flatten();

    pattStartPos = 0;
    pattern = flatedImg[pattStartPos];
    for i in range(1, len(flatedImg)):
        if (pattern == flatedImg[i]):
            continue
        compresedImg += list([i - pattStartPos, pattern]);
        pattStartPos = i;
        pattern = flatedImg[pattStartPos];
    compresedImg += list([len(flatedImg) - pattStartPos, pattern]);
    
    return compresedImg;

def decompressRLE(compresedImg):
    img = list();

    for i in range(0, len(compresedImg), 2):
        img += list([compresedImg[i + 1]] * int(compresedImg[i]));

    return img

def zigzag(A):
    template= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]

    return B

def quantize(data, Q, compressFlag):
    if (compressFlag):
        q = np.round(data / Q).astype(int)
        return q
    else:
        return data * Q

def compressBlock(block, Q):
    return zigzag(quantize(dct2(block), Q, True))

def compressLayer(layer, Q): 
    blocks = np.array([])
    for w in range(0, layer.shape[0], 8):
        for k in range(0, layer.shape[1], 8):
            block = layer[w:(w+8),k:(k+8)] - 128
            blocks = np.append(blocks, compressBlock(block,Q))

    return comressRLE(blocks)


def decompressBlock(vector, Q):
    return idct2(quantize(zigzag(vector), Q, False))

def decompressLayer(S, Q, height):
    S = np.array(S)
    L= np.zeros([height, S.shape[0] // height]) + 128
    for idx,i in enumerate(range(0,S.shape[0],64)):
        vector=S[i:(i+64)]
        m=L.shape[1]/8
        w=int((idx%m)*8)
        k=int((idx//m)*8)
        L[k:(k+8), w:(w+8)]= decompressBlock(vector,Q) + 128
    return L

def compress(img, tablesOfOneFlag, chromaResamplingType):
    jpeg = JPEG()
    jpeg.shape = img.shape

    if (not tablesOfOneFlag):
        jpeg.QY = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 36, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
        ])
        jpeg.QC = np.array([
           [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
        ])

    yCrCb = rgbToYCrCb(img);
    jpeg.Y =  yCrCb[:, :, 0]
    jpeg.Cr = yCrCb[:, :, 1]
    jpeg.Cb = yCrCb[:, :, 2]

    jpeg.ChromaRatio = chromaResamplingType

    y_before = jpeg.Y.shape[0] * jpeg.Y.shape[1]
    jpeg.Y = compressLayer(jpeg.Y, jpeg.QY)
    y_after = len(jpeg.Y)

    cr_before = jpeg.Cr.shape[0] * jpeg.Cr.shape[1]
    jpeg.Cr = compressLayer(chromaResampling(jpeg.Cr, jpeg.ChromaRatio), jpeg.QC)
    cr_after = len(jpeg.Cr)

    cb_before = jpeg.Cb.shape[0] * jpeg.Cb.shape[1]
    jpeg.Cb = compressLayer(chromaResampling(jpeg.Cb, jpeg.ChromaRatio), jpeg.QC)
    cb_after = len(jpeg.Cb)

    print("Compression:")
    print("Y layer: {}".format(y_before / y_after))
    print("Cr layer: {}".format(cr_before / cr_after))
    print("Cb layer: {}".format(cb_before / cb_after))

    return jpeg

def decompress(jpeg):
    jpeg.Y = decompressLayer(decompressRLE(jpeg.Y), jpeg.QY, jpeg.shape[0]).reshape(jpeg.shape[0], jpeg.shape[1])
    jpeg.Cr = np.array(chromaResamplingBack(decompressLayer(decompressRLE(jpeg.Cr), jpeg.QC, jpeg.shape[0]), jpeg.ChromaRatio, (jpeg.shape[0], jpeg.shape[1])))
    jpeg.Cb = np.array(chromaResamplingBack(decompressLayer(decompressRLE(jpeg.Cb), jpeg.QC, jpeg.shape[0]), jpeg.ChromaRatio, (jpeg.shape[0], jpeg.shape[1])))

    imgInYCrCb = np.dstack([jpeg.Y, jpeg.Cr, jpeg.Cb])
    return yCrCbToRgb(imgInYCrCb)
