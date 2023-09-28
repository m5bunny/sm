import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
import sklearn

def mean_squared_error(image1, image2):
    value = np.float64(0)
    image1 = image1.astype(np.int32)
    image2 = image2.astype(np.int32)
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            for z in range(image1.shape[2]):
                value += (np.abs(image1[x, y, z] - image2[x, y, z])) ** 2
    return value / (np.prod(image1.shape))

def normalized_mean_squared_error(image1, image2):
    return mean_squared_error(image1, image2) / mean_squared_error(image2, np.zeros(image2.shape))

def peak_signal_to_noise_ratio(image1, image2):
    return 10 * np.log10(255 ** 2 / mean_squared_error(image1, image2))

def mutual_information_factor(image1, image2):
    image1 = image1.astype(np.int32)
    image2 = image2.astype(np.int32)
    multiplication_sum = np.float64(0)
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            for z in range(image1.shape[2]):
                multiplication_sum += image1[x, y, z] * image2[x, y, z]

    sums = np.float64(0)
    for x in range(image1.shape[0]):
        for y in range(image1.shape[1]):
            for z in range(image1.shape[2]):
                sums += (image1[x, y, z] + image2[x, y, z]) ** 2

    return 1 - (sums / multiplication_sum)

if __name__ == "__main__":
    original_images = []
    broken_images = []

    for file in os.listdir("images"):
        original_images.append(plt.imread(f'images/{file}'))
    for file in os.listdir("images_results"):
        broken_images.append(plt.imread(f'images_results/{file}'))
    for index, filename in enumerate(os.listdir("results")):
        file_path = f'results/{filename}'
        dataframe = pd.read_csv(file_path, delimiter=';')

    original_image = original_images[index]
    broken_image_set = broken_images[index * 10:(index + 1) * 10]
    mse_scores = []
    nmse_scores = []
    psnr_scores = []
    mif_scores = []

    for i, broken_image in enumerate(broken_image_set):
        print(f'Processing Image #{i+1}')
        mse_scores.append(mean_squared_error(original_image, broken_image))
        nmse_scores.append(normalized_mean_squared_error(original_image, broken_image))
        psnr_scores.append(peak_signal_to_noise_ratio(original_image, broken_image))
        mif_scores.append(mutual_information_factor(original_image, broken_image))

    mse_scores = np.array(mse_scores)
    mse_scores = (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min())
    mse_scores = mse_scores * 5
    nmse_scores = np.array(nmse_scores)
    nmse_scores = (nmse_scores - nmse_scores.min()) / (nmse_scores.max() - nmse_scores.min())
    nmse_scores = nmse_scores * 5
    psnr_scores = np.array(psnr_scores)
    psnr_scores = (psnr_scores - psnr_scores.min()) / (psnr_scores.max() - psnr_scores.min())
    psnr_scores = psnr_scores * 5
    mif_scores = np.array(mif_scores)
    mif_scores = (mif_scores - mif_scores.min()) / (mif_scores.max() - mif_scores.min())
    mif_scores = mif_scores * 5

    fig, ax = plt.subplots()
    markers = cycle(['*', 'o', 's', '^', 'v', 'D', 'x', '+', 'P', 'h'])
    for i, row in dataframe.iterrows():
        scores = row.values[1:].astype(int)
        marker = next(markers)
        ax.scatter(range(1, len(scores) + 1), scores, marker=marker, label=f"Response #{i + 1}")

    ax.set_xlabel("# of Image")
    ax.set_ylabel("Score")
    ax.set_title("MOS")
    ax.set_xlim(0.5, 10.5)
    plt.margins(x=0.1)
    ax.legend()

    data = dataframe.values[:, 1:].astype(int)
    print(data.mean)
    linear_regression = np.polyfit(range(1, 11), data.mean(axis=0), 1)
    linear_regression_line = np.poly1d(linear_regression)
    plt.savefig(f'results_final/{filename}.png')
    plt.close()

    column_means = dataframe.iloc[:, 1:].mean()
    fig, ax = plt.subplots()

    ax.scatter(range(1, len(column_means) + 1), column_means, marker="o")
    ax.set_xlim(0.5, 10.5)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("# of Image")
    ax.set_ylabel("Mean")
    ax.set_title("MOS - mean by image")
    plt.savefig(f'results_final/{filename}-mean.png')
    plt.close()
        
    fig, ax = plt.subplots()
    markers = cycle(['*', 'o', 's', '^', 'v', 'D', 'x', '+', 'P', 'h'])
    for i, row in dataframe.iterrows():
        scores = row.values[1:].astype(int)
        marker = next(markers)
        ax.scatter(range(1, len(scores) + 1), scores, marker=marker, label=f"Response #{i + 1}")

    ax.set_xlabel("# of Image")
    ax.set_ylabel("Score")
    ax.set_title("MOS - Linear regression")
    ax.set_xlim(0.5, 10.5)
    plt.margins(x=0.1)

    ax.legend()
    data = dataframe.values[:, 1:].astype(int)
    linear_regression = np.polyfit(range(1, 11), data.mean(axis=0), 1)
    linear_regression_line = np.poly1d(linear_regression)
    plt.plot(range(1, 11), linear_regression_line(range(1, 11)), label="Linear regression")
    plt.savefig(f'results_final/{filename}-regression.png')
    plt.close()

    column_means = dataframe.iloc[:, 1:].mean()
    fig, ax = plt.subplots()
    ax.scatter(range(1, len(column_means) + 1), column_means, marker="o")
    ax.set_xlim(0.5, 10.5)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("# of Image")
    ax.set_ylabel("Mean")
    ax.set_title("MOS - mean by image - Linear regression")

    plt.plot(range(1, 11), linear_regression_line(range(1, 11)), label="Linear regression")
    plt.savefig(f'results_final/{filename}-mean-regression.png')
    plt.close()

    fig, ax = plt.subplots()
    markers = cycle(['*', 'o', 's', '^', 'v', 'D', 'x', '+', 'P', 'h'])
    for i, row in dataframe.iterrows():
        scores = row.values[1:].astype(int)
        marker = next(markers)
        ax.scatter(range(1, len(scores) + 1), scores, marker=marker, label=f"Response #{i + 1}")

    ax.set_xlabel("# of Image")
    ax.set_ylabel("Score")
    ax.set_title("MOS - MSE")

    ax.set_xlim(0.5, 10.5)
    plt.margins(x=0.1)
    ax.legend()
    plt.plot(range(1, 11), mse_scores, label="MSE")
    plt.savefig(f'results_final/{filename}-MSE.png')
    plt.close()

    column_means = dataframe.iloc[:, 1:].mean()

    fig, ax = plt.subplots()

    ax.scatter(range(1, len(column_means) + 1), column_means, marker="o")
    ax.set_xlim(0.5, 10.5)
    ax.set_xlim(0, 10.5)

    ax.set_xlabel("# of Image")
    ax.set_ylabel("Mean")
    ax.set_title("MOS - mean by image - MSE")
    plt.plot(range(1, 11), mse_scores, label="MSE")
    plt.savefig(f'results_final/{filename}-mean-MSE.png')
    plt.close()

    fig, ax = plt.subplots()
    markers = cycle(['*', 'o', 's', '^', 'v', 'D', 'x', '+', 'P', 'h'])
    for i, row in dataframe.iterrows():
        scores = row.values[1:].astype(int)
        marker = next(markers)
        ax.scatter(range(1, len(scores) + 1), scores, marker=marker, label=f"Response #{i + 1}")

    ax.set_xlabel("# of Image")
    ax.set_ylabel("Score")
    ax.set_title("MOS - NMSE")
    ax.set_xlim(0.5, 10.5)
    plt.margins(x=0.1)
    ax.legend()
    plt.plot(range(1, 11), nmse_scores, label="NMSE")
    plt.savefig(f'results_final/{filename}-NMSE.png')
    plt.close()

    column_means = dataframe.iloc[:, 1:].mean()

    fig, ax = plt.subplots()
    ax.scatter(range(1, len(column_means) + 1), column_means, marker="o")
    ax.set_xlim(0.5, 10.5)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("# of Image")
    ax.set_ylabel("Mean")
    ax.set_title("MOS - mean by image - NMSE")
    plt.plot(range(1, 11), nmse_scores, label="NMSE")
    plt.savefig(f'results_final/{filename}-mean-NMSE.png')
    plt.close()

    fig, ax = plt.subplots()
    markers = cycle(['*', 'o', 's', '^', 'v', 'D', 'x', '+', 'P', 'h'])
    for i, row in dataframe.iterrows():
        scores = row.values[1:].astype(int)
        marker = next(markers)
        ax.scatter(range(1, len(scores) + 1), scores, marker=marker, label=f"Response #{i + 1}")

    ax.set_xlabel("# of Image")
    ax.set_ylabel("Score")
    ax.set_title("MOS - PSNR")
    ax.set_xlim(0.5, 10.5)
    plt.margins(x=0.1)
    ax.legend()
    plt.plot(range(1, 11), psnr_scores, label="PSNR")
    plt.savefig(f'results_final/{filename}-PSNR.png')
    plt.close()

    column_means = dataframe.iloc[:, 1:].mean()
    fig, ax = plt.subplots()
    ax.scatter(range(1, len(column_means) + 1), column_means, marker="o")
    ax.set_xlim(0.5, 10.5)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("# of Image")
    ax.set_ylabel("Mean")
    ax.set_title("MOS - mean by image - PSNR")
    plt.plot(range(1, 11), psnr_scores, label="PSNR")
    plt.savefig(f'results_final/{filename}-mean-PSNR.png')
    plt.close()

    fig, ax = plt.subplots()
    markers = cycle(['*', 'o', 's', '^', 'v', 'D', 'x', '+', 'P', 'h'])
    for i, row in dataframe.iterrows():
        scores = row.values[1:].astype(int)
        marker = next(markers)
        ax.scatter(range(1, len(scores) + 1), scores, marker=marker, label=f"Response #{i + 1}")
    ax.set_xlabel("# of Image")
    ax.set_ylabel("Score")
    ax.set_title("MOS - IF")
    ax.set_xlim(0.5, 10.5)
    plt.margins(x=0.1)
    ax.legend()
    plt.plot(range(1, 11), m_if_scores, label="M-IF")
    plt.savefig(f'results_final/{filename}-IF.png')
    plt.close()

    column_means = dataframe.iloc[:, 1:].mean()
    fig, ax = plt.subplots()
    ax.scatter(range(1, len(column_means) + 1), column_means, marker="o")
    ax.set_xlim(0.5, 10.5)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel("# of Image")
    ax.set_ylabel("Mean")
    ax.set_title("MOS - mean by image - IF")
    plt.plot(range(1, 11), m_if_scores, label="M-IF")
    plt.savefig(f'results_final/{filename}-mean-IF.png')
    plt.close()
