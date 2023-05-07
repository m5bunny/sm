import numpy as np
import matplotlib.pyplot as plt

def colorFit(pixel, Pallet):
    return Pallet[np.argmin(np.linalg.norm(pixel-Pallet, axis = 1))]

def imgQuant(img, Pallet):
        out_img = img.copy()
        for i in range(img.shape[0]):
                for j  in range(img.shape[1]):
                        out_img[i][j]=colorFit(img[i][j],Pallet)
        return out_img

def imgToFloat(img):
    if (np.issubdtype(img.dtype,np.floating)):
        return img
    return img / 255.0

def getGrayScale1(img): 
    if (len(img.shape) < 3):
        return img;
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]


def randomDithering(img):
    grayscaleImg = getGrayScale1(img)
    randMap = np.random.uniform(low=0.0, high=1.0 , size=(grayscaleImg.shape[0], grayscaleImg.shape[1]))
    imgDithered = (grayscaleImg >= randMap) * 1;
    return imgDithered

def orderedDithering(img, Palette):
    bMatrix = np.array([[0,8,2,10],[12,4,14,6], [3,11,1,9], [15,7,13,5]])
    mp = 1 / (bMatrix.shape[0] * bMatrix.shape[1]) * (bMatrix + 1) - 0.5
    imgDithered = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgDithered[i][j] = colorFit(img[i][j] + mp[i % bMatrix.shape[0]][j % bMatrix.shape[1]], Palette)
    return imgDithered

def floydSteinbergDithering(img, Palette):
    imgDithered = img
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            old = img[i][j].copy()
            new = colorFit(old, Palette)
            imgDithered[i][j] = new 
            err = old - new 
            if (i < img.shape[0] - 1):
                imgDithered[i+1][j] = imgDithered[i+1][j] + err * 7/16
            if (j < img.shape[1] - 1):
                if (i > 0):
                    imgDithered[i-1][j+1] = imgDithered[i-1][j+1] + err * 3/16
                imgDithered[i][j+1] = imgDithered[i][j+1] + err * 5 / 16
                if (i < img.shape[0] - 1):
                    imgDithered[i+1][j+1] = imgDithered[i+1][j+1] + err * 1/16
    return imgDithered
