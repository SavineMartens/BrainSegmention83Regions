import os
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger, EarlyStopping
import time
from keras.optimizers import Adam
from keras.models import load_model, model_from_json
import keras.backend as K
import tensorflow as tf
from UNet3D import unet_3d_9_BN as unet
from UNet3D import *
from DatageneratorTest import DataGenerator
import nibabel as nib
from Shortlist import *
import matplotlib.pyplot as plt
from datetime import datetime
from Visualisation import *
from Configuration2 import *
import sys
import pylab as P
import glob


location = 'local'
sample_len = 30
it_val = 0.001  # learning rate

if location == 'cluster':
    training_type = sys.argv[1]  # "Baseline" / "DA_Hammers" / "DA_ADNI" / "BL_ADNI" / 'BL_Hammers' / 'BL_skull'
    lobe_pretrained = sys.argv[2]
    lobe_transfer = sys.argv[3]
    test_type = sys.argv[4]  # "ADNI" / 'Hammers'
    version = sys.argv[5]
    text_colour = 'black'
if location == 'local':
    training_type = 'ADNI_200' # "Baseline" / "DA_Hammers" / "DA_ADNI" / "BL_ADNI" / 'BL_Hammers' / 'BL_skull'
    lobe_pretrained = 'Temporal'
    lobe_transfer = 'Frontal'
    test_type = 'Hammers'  # "ADNI" / 'Hammers'
    version = '2a'
    text_colour = 'white'

print('Training type:', str(training_type), ' with learning rate', str(it_val), 'testing on', test_type)

train_path, test_path, save_path = path_per_type(location, training_type)
ref_shape = ref_shape_from_dir(test_path)
print('Reference shape:', ref_shape)

hyper_parameter_to_evaluate = 'Initial_LR'  # 'Batch_Size' / 'Alpha_ReLu' / 'TrainingSize'
parameter_save = os.path.join(save_path, training_type, 'Transfer' + lobe_pretrained + 'To' + lobe_transfer,
                              lateralisation + hyper_parameter_to_evaluate + 'E' + str(n_epoch))

model_save_path = os.path.join(parameter_save, 'Model' + version + 'Details' + str(it_val))
weight_path = os.path.join(model_save_path, 'model_weight.h5')
model_path = os.path.join(model_save_path, 'Vkeras_A1_model.hdf5')

areas = area_selection(lobe_transfer, lateralisation)
n_classes = len(areas) + 1
print('Transfer learning from', lobe_pretrained, 'lobe to', lobe_transfer, 'lobe')
print('Selecting areas from', lobe_transfer, 'lobe of', lateralisation, 'side(s)')
print('Number of classes including background:', n_classes)
print('Areas used in classification:', areas)

# image size Parameters
params = {'batch_size': num_batch,
          'dim_x': ref_shape[0],
          'dim_y': ref_shape[1],
          'dim_z': ref_shape[2],
          'n_classes': n_classes,
          'shuffle': True,
          'verbose': verbose_generator}

para_decay_auto = {'initial_lr': initial_lr,
                   'drop_percent': 0.5,
                   'patience': 15,
                   'threshold_epsilon': 0.0001,
                   'momentum': 0.8,
                   'nesterov': True}

train_end = sample_len - num_test - num_val
val_end = train_end + num_val
idx = np.arange(sample_len)

train_idx = idx[:train_end]
val_idx = idx[train_end:val_end]
predict_index = idx[val_end:]
test_index = predict_index

# Construct the model
print("Constructing model")
# unet_model = model_from_json(open(model_path).read()) #
# unet_model.load_weights(weight_path)
unet_model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef,
                                                    'val_dice_loss':dice_loss, 'val_dice_coef': dice_coef}) #unet(params['dim_x'], params['dim_y'], params['dim_z'], 1, nCommon_multiple, n_classes, alpha_relu)
Model_Summary = unet_model.summary()
# unet_model.load_weights(weight_path)
print("Loading pre-trained model from disk!")

# opt = Adam(lr=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# unet_model.compile(optimizer=opt, loss=[dice_loss], metrics=[dice_coef])

"""
test
"""

loss_overall = []
DC_all = []

if test_type == 'ADNI':
    print("Step 3: testing on control adni")
    if location == 'cluster':
        test_path = '/media/data/smartens/data/datasize176_208_160/ADNI/Test'
        test_list = np.load('/media/data/smartens/data/datasize176_208_160/ADNI/adni_test_set.npy')
        if 'skull' in training_type:
            test_path = '/media/data/smartens/data/datasize176_208_160/ADNI/TestSkullStripped'
    if location == 'local':
        test_path = '/home/jara/Savine/datasize176_208_160/ADNI/Test'
        if 'skull' in training_type:
            test_path = '/home/jara/Savine/datasize176_208_160/ADNI/TestSkullStripped'
        test_list = np.load('AdditionalFiles/adni_test_set.npy')
    pred_seg_path = os.path.join(parameter_save, 'PredictedSegmentation' + str(training_type) + str(it_val))
    if not os.path.exists(pred_seg_path):
        os.makedirs(pred_seg_path)
    if glob.glob(pred_seg_path + '/worst*.npy'):
        worst_list = np.load(glob.glob(pred_seg_path + '/worst*.npy')[0])
        best_list = np.load(glob.glob(pred_seg_path + '/best*.npy')[0])
        selection_segmentation = list(worst_list) + list(best_list)
    print('Length ADNI test set:', len(test_list))

    for i, idx in enumerate(test_list):
        print(i, idx)
        if 'skull' in training_type:
            X_i = nib.load("".join(test_path + '/' + idx + 'cropbrain.nii.gz'))
            Y_i = nib.load("".join(test_path + '/' + idx + 'cropseg.nii.gz'))
        else:
            X_i = nib.load("".join(test_path + '/' + idx + '_blcropbrain.nii.gz'))
            Y_i = nib.load("".join(test_path + '/' + idx + '_blcropseg.nii.gz'))
        X_i = np.asarray(X_i.get_data())
        X_i = np.expand_dims(X_i, axis=4)
        Y_i = np.asarray(Y_i.get_data())
        Y_i = remove_areas(Y_i, areas)
        Y_i = encode1hot(Y_i)
        dimx = np.expand_dims(X_i, axis=0)
        dimy = np.expand_dims(Y_i, axis=0)
        results_i = unet_model.evaluate(dimx, dimy, batch_size=1)
        print(results_i)

        # make segmented images
        if glob.glob(pred_seg_path + '/worst*.npy'):
            if idx in selection_segmentation:
                seg_images = unet_model.predict(np.expand_dims(X_i, 0), batch_size=1)
                Y_pred_data = np.argmax(seg_images, axis=-1)

                show_differences(X_i, Y_i, Y_pred_data, results_i[1])
                if idx in worst_list:
                    plt.savefig(os.path.join(pred_seg_path, 'DifferenceADNI' + idx + 'bestDice' +
                                             str(round(results_i[1], 3)) + '.png'), format='png')
                if idx in best_list:
                    plt.savefig(os.path.join(pred_seg_path, 'DifferenceADNI' + idx + 'worstDice' +
                                             str(round(results_i[1], 3)) + '.png'), format='png')
        # else:
        #     if i//5 == 0:
        #         seg_images = unet_model.predict(np.expand_dims(X_i, 0), batch_size=1)
        #         Y_pred_data = np.argmax(seg_images, axis=-1)
        #         plt.savefig(os.path.join(pred_seg_path, 'DifferenceADNI' + idx + 'Dice' + str(round(results_i[1], 3))),
        #             format='png')
        loss_overall.append(results_i[0])
        DC_all.append(results_i[1])

if test_type == 'Hammers':
    print("Step 3: testing on test set Hammers")
    test_index = [26, 27, 28, 29]
    pred_seg_path = os.path.join(parameter_save, 'PredictedSegmentation' + str(training_type) + str(it_val))
    if not os.path.exists(pred_seg_path):
        os.makedirs(pred_seg_path)

    for idx in test_index:
        ID = str(idx + 1)
        ID = ID.zfill(2)
        X_i = nib.load("".join(test_path + '/a' + ID + '.nii.gz'))
        X_i = np.asarray(X_i.get_data())
        X_i = np.expand_dims(X_i, axis=4)
        Y_i = nib.load("".join(test_path + '/a' + ID + '-seg.nii.gz'))
        Y_i = np.asarray(Y_i.get_data())
        Y_i = remove_areas(Y_i, areas)
        Y_i = encode1hot(Y_i)
        dimx = np.expand_dims(X_i, axis=0)
        dimy = np.expand_dims(Y_i, axis=0)
        results_i = unet_model.evaluate(dimx, dimy, batch_size=1)
        print(results_i)

        seg_images = unet_model.predict(np.expand_dims(X_i, 0), batch_size=1)
        Y_pred_data = np.argmax(seg_images, axis=-1)

        time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_stamp = time_stamp.replace(" ", "_")
        show_differences(X_i, Y_i, Y_pred_data, results_i[1], text_colour)
        plt.savefig(os.path.join(pred_seg_path, 'DifferenceHammers' + ID + 'Dice' + str(round(results_i[1], 3))), format='png')
        loss_overall.append(results_i[0])
        DC_all.append(results_i[1])


results = [np.mean(DC_all), sum(loss_overall)/len(loss_overall)]
print('Dice coefficient test data:', results[0])
print('Dice coefficient standard deviation', np.std(DC_all))

np.save(os.path.join(pred_seg_path, 'DiceCoefficientResultsfor' + test_type + '.npy'), DC_all, allow_pickle=True)