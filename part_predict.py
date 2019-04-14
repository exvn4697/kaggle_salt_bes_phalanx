import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from params import args


# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def read_phalanx_test(path):
    phalanx = np.load(path)

    min_prob = float("inf")
    max_prob = -float("inf")
    for m in phalanx:
        mi = np.min(m)
        ma = np.max(m)
        if mi < min_prob:
            min_prob = mi
        if ma > max_prob:
            max_prob = ma
    
    test_id = [x[:-4] for x in os.listdir('/test_data') if x[-4:] == '.png']

    phalanx_dict = {}
    for idx, val in enumerate(test_id):
        phalanx_dict[val] = (phalanx[idx] - min_prob) / (max_prob - min_prob)

    return phalanx_dict

if __name__ == '__main__':
    test_ids = [x[:-4] for x in os.listdir('/test_data') if x[-4:] == '.png']
    
    rles = []
    pred = read_phalanx_test('/workdir/phalanx/weights/model_256_res34v402.pth')
    threshold = 0.5
    for img_id in tqdm(test_ids):
        mask = pred[img_id] > threshold
        rle = RLenc(mask)
        rles.append(rle)
    
    test = pd.DataFrame({'id': test_ids, 'rle_mask': rles})

    test[['id', 'rle_mask']].to_csv('/', index=False)