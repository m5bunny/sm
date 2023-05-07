import numpy as np
import matplotlib.pyplot as plt
import cv2


def weight_mean(fragment, axis):
    weights = np.empty(fragment.shape)
    for i in range(0, fragment.shape[0]):
        weights[i] = np.mean(fragment[i]).astype('int')
    fragment = np.multiply(fragment, weights).astype('int');
    fragment_sums = np.sum(fragment, axis)
    weights_sum = np.sum(weights)
    return fragment_sums / weights_sum


def scale(img, scale_factor, method):
    if (scale_factor == 1):
        return img;

    Y, X, DIMENSION = img.shape
    
    out_y = round(Y * scale_factor)
    out_x = round(X * scale_factor)

    out_img = np.empty([out_y, out_x, DIMENSION]).astype('uint8')
    out_y_i = np.linspace(0, Y - 1, out_y)
    out_x_i = np.linspace(0, X - 1, out_x)

    # Zoom in
    if (scale_factor > 1):
        if (method == 'nearest'):
            for i in range(0, out_y):
                input_i = round(out_y_i[i])
                for j in range(0, out_x):
                    input_j = round(out_x_i[j])
                    out_img[i][j] = img[input_i][input_j].copy();

        elif (method == 'interpolation'):
            for i in range(0, out_y):
                input_i = round(out_y_i[i])
                for j in range(0, out_x):
                    input_j = round(out_x_i[j])
                    x1 = int(max(min(np.floor(input_j), X - 1), 0))
                    x2 = int(max(min(np.ceil(input_j), X - 1), 0))
                    y1 = int(max(min(np.floor(input_i), Y - 1), 0))
                    y2 = int(max(min(np.ceil(input_i), Y - 1), 0))
                    q11 = img[y1][x1]
                    q12 = img[y2][x1]
                    q21 = img[y1][x2]
                    q22 = img[y2][x2]
                    dx = input_j - x1
                    dy = input_i - y1

                    out_img[i][j] = q11 * (1 - dx) * (1 - dy)
                    out_img[i][j] += q12 * dy * (1 - dx)
                    out_img[i][j] += q21 * dx * (1 - dy)
                    out_img[i][j] += q22 * dx * dy
        else: 
            raise Exception("Unsupported zoom in method")


    # Zoom out
    if (scale_factor < 1):
        RANGE = 3
        y_sur_indexes = np.empty([out_y, 2]).astype('int')
        x_sur_indexes = np.empty([out_x, 2]).astype('int')

        func = None
        if (method == 'mean'):
            func = np.mean
        elif (method == 'median'):
            func = np.median
        elif (method == 'weight'):
            func = weight_mean
        else: 
            raise Exception("Unsupported zoom out method")


        for i in range(0, out_y):
            start = np.round(out_y_i[i]) - RANGE
            end = np.round(out_y_i[i]) + RANGE
            y_sur_indexes[i] = [start if start > 0 else 0, end if end < Y else Y]

        for i in range(0, out_x):
            start = np.round(out_x_i[i]) - RANGE
            end = np.round(out_x_i[i]) + RANGE
            x_sur_indexes[i] = [start if start > 0 else 0, end if end < X else X]


        for i in range(0, out_y):
            for j in range(0, out_x):
                sur = img[y_sur_indexes[i][0]:y_sur_indexes[i][1],x_sur_indexes[j][0]:x_sur_indexes[j][1]]
                sur_x, sur_y, sur_dim = sur.shape
                out_img[i][j] = func(np.reshape(sur, (sur_x * sur_y, sur_dim)), axis=0)

    return out_img
