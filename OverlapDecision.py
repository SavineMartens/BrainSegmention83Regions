import numpy as np
from UNet3D import unet_3d_9_BN as unet
import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger, EarlyStopping
import time
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from UNet3D import unet_3d_9_BN as unet
from UNet3D import *
from Datagenerator import DataGenerator
import nibabel as nib
from Shortlist import *
import matplotlib.pyplot as plt
from datetime import datetime
from Visualisation import *
from Configuration import *
import sys
import pylab as P
import glob
import random


class Overlap(object):
    def __init__(self, Y1, Y2, overlap):
        'Initialization'
        self.Y1 = Y1  # new_labels
        self.Y2 = Y2  # old labels
        self.overlap = overlap

    def majority_vote(self, cube_size):
        print('Using majority for overlapping voxels')
        d = cube_size
        e = d - 2

        remove_overlap1 = np.where(self.overlap == 1, 0, self.Y1)
        remove_overlap2 = np.where(self.overlap == 1, 0, self.Y2)
        merged_Y = remove_overlap1 + remove_overlap2
        merged_Y = np.where(self.overlap == 1, np.nan, merged_Y)
        updated_overlap = self.overlap
        n_overlap = int(np.sum(updated_overlap))
        location = np.where(updated_overlap == 1)
        count = 0
        for n in np.arange(n_overlap):
            x, y, z = location[0][n], location[1][n], location[2][n]

            # select dxdxd cube for decision
            cube = merged_Y[x-e:x+e+1, y-e:y+e+1, z-e:z+e+1]
            (labels, counts) = np.unique(cube, return_counts=True)
            # sort it lowest to highest
            sorted_counts, sorted_labels = (list(t) for t in zip(*sorted(zip(counts, labels))))

            if sorted_counts[-1] > sorted_counts[-2]:
                label = sorted_labels[-1]
            if sorted_counts[-1] == sorted_counts[-2]:
                label = sorted_labels[-1-random.randint(0, 1)]
                count+=1
            merged_Y[x, y, z] = label
            updated_overlap[x, y, z] = 0

        print('Tie in', round(count/n_overlap*100,3), '%')
        return merged_Y

    def whole_brain_network(self, X, weight_path):
        print('Using whole brain network for overlapping voxels')
        ref_shape = X.shape
        n_classes = 9
        n_overlap = int(np.sum(self.overlap))

        # image size Parameters
        params = {'batch_size': 1,
                  'dim_x': ref_shape[0],
                  'dim_y': ref_shape[1],
                  'dim_z': ref_shape[2],
                  'n_classes': n_classes,
                  'shuffle': True,
                  'verbose': False}

        para_decay_auto = {'initial_lr': initial_lr,
                           'drop_percent': 0.5,
                           'patience': 15,
                           'threshold_epsilon': 0.0001,
                           'momentum': 0.8,
                           'nesterov': True}

        print("Constructing whole brain model")
        unet_model = unet(params['dim_x'], params['dim_y'], params['dim_z'], 1, nCommon_multiple, n_classes, alpha_relu)
        unet_model.load_weights(weight_path)
        print("Loading pre-trained model from", weight_path)

        opt = Adam(lr=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        unet_model.compile(optimizer=opt, loss=[dice_loss_all], metrics=[dice_coef_prob])

        seg_images = unet_model.predict(np.expand_dims(X, 0), batch_size=1)
        Y_pred_data = np.squeeze(np.argmax(seg_images, axis=-1))
        location = np.where(self.overlap == 1)

        remove_overlap1 = np.where(self.overlap == 1, 0, self.Y1)
        remove_overlap2 = np.where(self.overlap == 1, 0, self.Y2)
        merged_Y = remove_overlap1 + remove_overlap2
        merged_Y = np.where(self.overlap == 1, np.nan, merged_Y)

        for n in np.arange(n_overlap):
            x, y, z = location[0][n], location[1][n], location[2][n]
            label1 = self.Y1[x, y, z]
            label2 = self.Y2[x, y, z]
            label_whole_brain = Y_pred_data[x, y, z]
            areas_in_lobe = lobes_to_areas(label_whole_brain)
            if areas_in_lobe == 0:
                merged_Y[x, y, z] = 0  # belongs to background
            elif label1 in areas_in_lobe:
                merged_Y[x, y, z] = label1
            elif label2 in areas_in_lobe:
                merged_Y[x, y, z] = label2
            elif label1 not in areas_in_lobe and label2 not in areas_in_lobe:
                # when it belongs to a lobe that has not been processed yet
                merged_Y[x, y, z] = 0  # it will hopefully be assigned later when that lobe is assessed
            else:
                Y1 = self.Y1
                Y2 = self.Y2
                print('Something else is going on and I don\'t know what')
        print('Finished labelling overlapping voxels')
        if np.isnan(merged_Y).any():
            print('Still NaN in merged Y')

        del unet_model  # will speed up process?

        return merged_Y

