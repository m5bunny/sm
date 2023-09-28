import os
import cv2
import numpy as np

def jpeg(filename):
    name = os.path.splitext(os.path.split(filename)[1])[0]
    img = cv2.imread(filename)
    for i in range(1, 11):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), i*7]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        cv2.imwrite(name+"_" + str(i) + "_compress" + ".jpg", decimg)


def blur(filename):
    name = os.path.splitext(os.path.split(filename)[1])[0]
    nameExt = os.path.splitext(os.path.split(filename)[1])[1]

    img = cv2.imread(filename)
    for i in range(1, 11):
        kernal_size = (i*5, i*5)
        blur_img = cv2.blur(img, kernal_size)
        cv2.imwrite(name + "_" + str(i) + "_blur" + nameExt, blur_img)


def noise(filename):
    name = os.path.splitext(os.path.split(filename)[1])[0]
    nameExt = os.path.splitext(os.path.split(filename)[1])[1]

    img = cv2.imread(filename)
    img = img.astype(np.int32)

    for i in range(1, 11):
        alpha = 0.1 * i

        sigma = 25
        gauss = np.random.normal(0, sigma, (img.shape))
        noisy1 = (img + alpha * gauss).clip(0, 255).astype(np.uint8)
        cv2.imwrite(name + "_" + str(i) + "_noise" + nameExt, noisy1)

def main():
    jpeg('./jpeg.jpg')
    blur('blur.jpg')
    noise('noise.jpg')

if __name__ == "__main__":
    main()
