from typing import Counter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj,np.ndarray):
        size=obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def convertToUint8(img):
    if(img.dtype == 'uint8'):
        return img;
    else:
        return (img*255).astype(np.uint8);


def comressRLE(img):
    compresedImg = [len(img.shape)];
    compresedImg += list(img.shape);
    flatedImg = img.flatten();

    pattStartPos = 0;
    pattern = flatedImg[pattStartPos];
    for i in tqdm(range(1, len(flatedImg))):
        if (pattern == flatedImg[i]):
            continue
        compresedImg += list([i - pattStartPos, pattern]);
        pattStartPos = i;
        pattern = flatedImg[pattStartPos];
    compresedImg += list([len(flatedImg) - pattStartPos, pattern]);

    return compresedImg;

def decompressRLE(compresedImg):
    img = list();

    for i in tqdm(range(compresedImg[0] + 1, len(compresedImg), 2)):
        img += list([compresedImg[i + 1]] * compresedImg[i]);

    return np.array(img).reshape(compresedImg[1:int(compresedImg[0] + 1)])


def compressByteRun(img):
    compresedImg = [len(img.shape)];
    compresedImg += list(img.shape);
    flatedImg = img.flatten();
    
    i = 0
    while i < len(flatedImg):
        pattern = flatedImg[i]
        count = 1

        for j in range(i + 1, min(i + 128, len(flatedImg))):
            if flatedImg[j] == pattern:
                count += 1
            else:
                break
        if count > 1:
            compresedImg += list([-(count - 1), pattern]);
            i += count
            continue;
        else:
            for j in range(i + 1, min(i + 128, len(flatedImg))):
                if (pattern == flatedImg[j]):
                    count -= 1;
                    break;
                count += 1;
                pattern = flatedImg[j];

            compresedImg += list([(count - 1)]);
            compresedImg += list(flatedImg[i:i+count]);
            i += count;

    return compresedImg 

def decompressByteRun(compresedImg):
    img = list()
    i = compresedImg[0] + 1;
    while i < len(compresedImg):
        count = compresedImg[i]
        if count < 0:
            count -= 1
            img += list([compresedImg[i+1]] * (-count));
            i += 2;
        else:
            count += 1;
            i += 1;
            img += list(compresedImg[i:i + count])
            i += count;

    return np.array(img).reshape(compresedImg[1:int(compresedImg[0] + 1)])

imgNames = ['./tech.jpg', './doc.jpg', './col.jpg']

for imgName in imgNames:
    img = convertToUint8(plt.imread(imgName));
    imgSize = get_size(img);

    rltCompresed = comressRLE(img)
    rltDecompresed = decompressRLE(rltCompresed)
    rltCSize = get_size(rltCompresed);
    rltDSize = get_size(rltDecompresed)

    byteRunCompresed = compressByteRun(img)
    byteRunDecompresed = decompressByteRun(byteRunCompresed)
    brCSize = get_size(byteRunCompresed);
    brDSize = get_size(byteRunDecompresed);

    print('Original size: {}, RLT decompresed size: {}, Byte run decompresed size: {}'.format(imgSize, rltDSize, brDSize))
    print('RLT compresed size: {}, Byte run compresed size: {}'.format(rltCSize, brCSize))
    print('RLT copmression ratio: {}, Byte run compression ratio: {}'.format(round(imgSize / rltCSize, 2), round(imgSize / brCSize, 2)))
    print('RLT % of original size: {}, Byte run % of original size: {}'.format(round(rltCSize / imgSize, 2) * 100, round(brCSize / imgSize, 2) * 100))

    plt.subplot(1, 3, 1)
    plt.title('Reference')
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title('Compresed and decompresed by RLT')
    plt.imshow(rltDecompresed)

    plt.subplot(1, 3, 3)
    plt.title('Compresed and decompresed by Byte Run')
    plt.imshow(byteRunDecompresed)

    plt.show()

