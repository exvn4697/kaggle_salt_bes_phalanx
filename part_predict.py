import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import argparse
import sys

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--epochs', type=int, default=50)
arg('--n_snapshots', type=int, default=1)
arg('--fold', default='0')
arg('--pretrain_weights')
arg('--prediction_weights', default='fold_{}.hdf5')
arg('--prediction_folder', default='oof')
arg('--learning_rate', type=float, default=0.0001)
arg('--input_size', type=int, default=192)
arg('--resize_size', type=int, default=160)
arg('--batch_size', type=int, default=24)

arg('--loss_function', default='bce_jacard')
arg('--augmentation_name', default='valid')
arg('--augmentation_prob', type=float, default=1.0)
arg('--network', default='unet_resnet_50')
arg('--alias', default='')
arg('--callback', default='snapshot')
arg('--freeze_encoder', type=int, default=0)

arg('--models_dir', default='/workdir/bes/weights/')
arg('--data_root', default='/workdir/data/')
arg('--images_dir', default='/workdir/data/train/images/')
arg('--pseudolabels_dir', default='')
arg('--masks_dir', default='/workdir/data/train/masks/')
arg('--test_folder', default='/test_data/')
arg('--folds_csv', default='/workdir/data/train_proc_v2_gr.csv')
arg('--pseudolabels_csv', default='/workdir/data/pseudolabels_confident.csv')

arg('--initial_size', type=int, default=101)
arg('--num_workers', type=int, default=12)
arg('--early_stop_patience',  type=int, default=15)
arg('--reduce_lr_factor',  type=float, default=0.25)
arg('--reduce_lr_patience',  type=int, default=7)
arg('--reduce_lr_min',  type=float, default=0.000001)

arg('--stage',  type=int, default=3)
arg('--postprocessing',  type=int, default=0)
arg('--test_predictions_path', default='/workdir/predictions/test_predictions.csv')

args = parser.parse_args()



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
    pred = read_phalanx_test('/workdir/phalanx/fold_predictions/res34v4_pred0.npy')
    threshold = 0.5
    for img_id in tqdm(test_ids):
        mask = pred[img_id] > threshold
        rle = RLenc(mask)
        rles.append(rle)
    
    test = pd.DataFrame({'id': test_ids, 'rle_mask': rles})

    test[['id', 'rle_mask']].to_csv('/workdir', index=False)