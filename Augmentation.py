import numpy as np
import matplotlib.pyplot as plt
import os
import math
from nilearn import image
import nibabel as nib
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import keras as K
from datetime import datetime
from Visualisation import *
from keras.models import load_model
from Shortlist import *
from keras.utils import np_utils
import random

#############################################################################################################
path = '/media/data/smartens/data/datasize176_208_160/Hammers'
sample_len = 26
new_data_per_sample = 25

range_rot = 10
range_shear = 5
range_trans = 5
range_zoom = 0.05
flip_ratio = 0.7  # lower number means higher chance of flipping
bool_save = True

#############################################################################################################
save_path = '/media/data/smartens/data/datasize176_208_160/HammersAugmented'  #"".join(path + '/AugmentedData_' + str(new_data_per_sample) + 'PerSample')
if not os.path.exists(save_path):
    os.makedirs(save_path)

datagen = ImageDataGenerator()

if bool_save:
    text_path = "".join(save_path + '/TransformationParameters.txt')
    file = open(text_path, 'w+')
    text0 = "".join("Range of transformation parameters" + "\n")
    text1 = "".join("Rotation: " + str(range_rot) + "\n")
    text2 = "".join("Shear: " + str(range_shear) + "\n")
    text3 = "".join("Translation_shift: " + str(range_trans) + "\n")
    text4 = "".join("Zoom: " + str(range_zoom) + "\n")
    text5 = "".join("Flip_ratio: " + str(flip_ratio) + "\n\n")
    text = [text0, text1, text2, text3, text4, text5]
    for t, line in enumerate(text):
        file.writelines(line)
    file.close()

# rr1 = np.empty((new_data_per_sample, 160, 144, 80))
# oo2 = np.empty((new_data_per_sample, 160, 144, 80))


for i in np.arange(sample_len):
    ID = str(i+1)
    print('Data augmentation for original image:', ID)
    ID = ID.zfill(2)
    xname = "".join('a' + ID + '.nii.gz')
    x_path = os.path.join(path, xname)
    X_temp = nib.load(x_path)
    X_temp = np.asarray(X_temp.get_data())

    yname = "".join('a' + str(ID) + '-seg.nii.gz')
    y_path = os.path.join(path, yname)
    Y_temp = nib.load(y_path)
    Y_temp = np.asarray(Y_temp.get_data())

    for j in np.arange(new_data_per_sample):
        print('Augmented image ', j+1, '/', new_data_per_sample)
        tuple = X_temp.shape + (1,)
        X = np.empty(tuple)
        Y = np.empty(tuple, dtype=int)

        rand_angles = [random.uniform(-range_rot, range_rot), random.uniform(-range_shear, range_shear)]
        rand_trans = np.random.uniform(-range_trans, range_trans, 2)
        rand_zoom = np.random.uniform(1-range_zoom, 1+range_zoom, 2)
        rand_flip = random.random() > flip_ratio

        transform_parameters = {'theta': rand_angles[0],
                                'tx': rand_trans[0],
                                'ty': rand_trans[1],
                                'shear': rand_angles[1],
                                'zx': rand_zoom[0],
                                'zy': rand_zoom[1],
                                'flip_vertical': rand_flip  # is actually vertical w.r.t. reality
                                }

        for s in range(X_temp.shape[1]):
            print(s)
            X_slice = X_temp[:, s, ]
            X_slice = np.expand_dims(X_slice, axis=3)  # needs to be 3D
            X[:, s, ] = datagen.apply_transform(X_slice, transform_parameters)
            print('x: ', transform_parameters)

            Y_slice = Y_temp[:, s, ]
            Y_slice = np.expand_dims(Y_slice, axis=3)
            Y[:, s, ] = datagen.apply_transform(Y_slice, transform_parameters)
            print('y: ', transform_parameters)

            # if s == 80:
            #     # fig = plt.figure()
            #     # plt.imshow(np.squeeze(X_slice), cmap="gray")
            #     # plt.imshow(np.squeeze(Y_slice), alpha=0.3)
            #     # plt.show()
            #     fig = plt.figure()
            #     plt.imshow(np.squeeze(X[:, s, ]), cmap="gray")
            #     plt.imshow(np.squeeze(Y[:, s, ]), alpha=0.3)
            #     plt.show()

        # r = np.squeeze(X)
        # o = np.squeeze(Y)
        #
        # rr1[j, ] = r
        # oo1[j, ] = o
        # show_slices([r[28, :, :], r[:, 33, :], r[:, :, 28]], "gray")
        # show_slices([o[28, :, :], o[:, 33, :], o[:, :, 28]], "pink")

        if bool_save:
            # Saving transformation per generated image
            file = open(text_path, 'a+')
            text0 = "".join("Augmented image "+ str(j) + " of original image " + ID + "\n")
            text1 = "".join('theta: ' + str(rand_angles[0]) + "\n")
            text2 = "".join('tx: ' + str(rand_trans[0]) + "\n")
            text3 = "".join('ty: ' + str(rand_trans[1]) + "\n")
            text4 = "".join('shear: ' + str(rand_angles[1]) + "\n")
            text5 = "".join('zx: ' + str(rand_zoom[0]) + "\n")
            text6 = "".join('zy: ' + str(rand_zoom[1]) + "\n")
            text7 = "".join('flip_horizontal: ' + str(rand_flip) + "\n\n")
            text = [text0, text1, text2, text3, text4, text5, text6, text7]
            for t, line in enumerate(text):
                file.writelines(line)
            file.close()

            # Saving augmented data as numpy array in tf tensor shape [x, y, z, 1]
            x_str = "".join(ID + '_' + str(j+1).zfill(2))
            y_str = "".join(x_str + '-seg')
            # np.save(os.path.join(save_path, x_str), X, allow_pickle=True)
            # np.save(os.path.join(save_path, y_str), Y, allow_pickle=True)

            img = nib.Nifti1Image(np.squeeze(X), np.eye(4, 4))
            nib.save(img, os.path.join(save_path, 'a' + x_str + '.nii.gz'))

            labels = nib.Nifti1Image(np.squeeze(Y), np.eye(4, 4))
            nib.save(labels, os.path.join(save_path, 'a' + y_str + '.nii.gz'))


# max_slice_of_planes_viewer_mc(rr1, oo1, 'True')

list_training = []
for i in range(1, 23):
    first = str(i).zfill(2)
    for j in range(1, new_data_per_sample+1):
        second = str(j).zfill(2)
        index = "".join(first + '_' + second)
        list_training.append(index)

np.save(os.path.join(save_path, 'ListTraining'), list_training)

