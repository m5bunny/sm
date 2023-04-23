import numpy as np
import matplotlib.pyplot as plt
import cv2

def imgToFloat(img):
    if (np.issubdtype(img.dtype,np.floating)):
        return img
    return img / 255.0

def imgToUInt8(img):
    if (np.issubdtype(img.dtype,np.unsignedinteger)):
        return img
    return (img * 255).astype('uint8')


def getGrayScale1(img): 
    if (len(img.shape) < 3):
        return img;
    return 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

def getGrayScale2(img): 
    return 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    

def zad1():
    img = imgToFloat(plt.imread('./image.jpg'))

    plt.imshow(img)
    plt.axis("off")
    plt.show()

    plt.imshow(getGrayScale1(img), cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()

    plt.imshow(getGrayScale2(img), cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()

def zad2():
    img = imgToFloat(plt.imread('./B01.png'))

    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(getGrayScale1(img), cmap=plt.cm.gray)
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.imshow(getGrayScale2(img), cmap=plt.cm.gray)
    plt.axis("off")

    plt.subplot(3, 3, 4)
    plt.imshow(img[:,:,0], cmap=plt.cm.gray)
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.imshow(img[:,:,1], cmap=plt.cm.gray)
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.imshow(img[:,:,2], cmap=plt.cm.gray)
    plt.axis("off")

    plt.subplot(3, 3, 7)
    rImg = img.copy()
    rImg[:,:,1] = rImg[:,:,2] = 0
    plt.imshow(rImg)
    plt.axis("off")

    plt.subplot(3, 3, 8)
    gImg = img.copy()
    gImg[:,:,0] = gImg[:,:,2] = 0
    plt.imshow(gImg)
    plt.axis("off")

    plt.subplot(3, 3, 9)
    bImg = img.copy()
    bImg[:,:,0] = bImg[:,:,1] = 0
    plt.imshow(bImg)
    plt.axis("off")

    plt.show()
    
def zad3():
    img = imgToFloat(plt.imread('./B02.jpg'))

    fragsBorders = [[0, 200, 0, 200], [200, 400, 200, 400],[400, 600, 400, 600]];
    
    print(img.shape)
    for i, borders in enumerate(fragsBorders, start=1):
        fragment = img[borders[0]:borders[1], borders[2]:borders[3]].copy()
        plt.imshow(fragment)
        plt.axis("off")
        plt.show()
        plt.savefig("fragment{0}.png".format(i))

zad2()
