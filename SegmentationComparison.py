import os
import numpy as np
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
import glob
from Metrics import *
from Visualisation2 import *

test_path = '/home/jara/Savine/datasize176_208_160/Hammers_skull'
test_idx = [26, 27, 28, 29]

pred_path = '/home/jara/Savine/datasize176_208_160/hyperparametertuning/ADNIskull_300a/FrontalLeft/Initial_LRE200'

list_pred = glob.glob(pred_path + '/PredictedSegmentationADNIskull_300a0.001/Y_pred*')
ref_shape = [176, 208, 160]
# image size Parameters
params = {'batch_size': num_batch,
          'dim_x': ref_shape[0],
          'dim_y': ref_shape[1],
          'dim_z': ref_shape[2],
          'n_classes': 13,
          'shuffle': True,
          'verbose': verbose_generator}

para_decay_auto = {'initial_lr': initial_lr,
                   'drop_percent': 0.5,
                   'patience': 15,
                   'threshold_epsilon': 0.0001,
                   'momentum': 0.8,
                   'nesterov': True}

for i, idx in enumerate(test_idx):
    areas = [70] #lobe_selection('FrontalLeft')
    ID = str(idx + 1)
    ID = ID.zfill(2)
    X_i = nib.load("".join(test_path + '/a' + ID + '.nii.gz'))
    X_i = np.asarray(X_i.get_data())
    X_i = np.expand_dims(X_i, axis=4)
    Y_i = nib.load("".join(test_path + '/a' + ID + '-seg.nii.gz'))
    Y_true_all = np.asarray(Y_i.get_data())
    Y_true_sel = remove_areas(Y_true_all, areas)
    Y_i = encode1hot(Y_true_sel)

    indices = AnatomicalPlaneViewer(np.squeeze(X_i), Y_true_sel, Y_true_sel).max_of_slice('gt', 1)
    plt.show()

    Y_pred = np.squeeze(np.load(list_pred[i]))
    DC_own, DC_classes, DC_foreground = DiceCoefficientIndividualClass(np.squeeze(Y_i), encode1hot(Y_pred))
    # DC_own, DC_classes_i = Metrics(Y_true_sel, Y_pred).DSC()

    Y_true_new = np.zeros(Y_true_sel.shape)
    areas_incl_BG = areas.copy()
    areas_incl_BG.insert(0, 0)
    for i, item in enumerate(areas_incl_BG):
        Y_true_new = np.where(Y_true_sel==item, i, Y_true_new)

    AnatomicalPlaneViewer(np.squeeze(X_i), Y_pred, Y_pred).max_of_slice('prediction', DC_own)
    plt.show()

    AnatomicalPlaneViewer(np.squeeze(X_i), Y_true_new, Y_pred).show_differences(DC_own, False)
    plt.show()

    print("Constructing model")
    unet_model = unet(params['dim_x'], params['dim_y'], params['dim_z'], 1, nCommon_multiple, params['n_classes'], alpha_relu)
    Model_Summary = unet_model.summary()
    unet_model.load_weights(os.path.join(pred_path, 'ModelDetails0.001/model_weight.h5'))
    print("Loading pre-trained model")

    opt = Adam(lr=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    unet_model.compile(optimizer=opt, loss=[dice_loss], metrics=[dice_coef_prob])
    dimx = np.expand_dims(X_i, axis=0)
    dimy = np.expand_dims(Y_i, axis=0)
    results_i = unet_model.evaluate(dimx, dimy, batch_size=1)

    seg_images = unet_model.predict(np.expand_dims(X_i, 0), batch_size=1)
    Y_pred_data = np.argmax(seg_images, axis=-1)
    print(results_i)

    APV_params = {'X': np.squeeze(X_i),
                  'Y_true': Y_true_new.copy(),
                  'Y_pred': Y_pred.copy()}
    AnatomicalPlaneViewer(**APV_params).show_differences(results_i[1], False)
    plt.show()

    print('Next volume')