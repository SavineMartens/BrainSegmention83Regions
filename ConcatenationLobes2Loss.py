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
from OverlapDecision import Overlap
from Metrics import *
import sys
import time
import csv

lobe_list = ['TemporalLeft', 'TemporalRight', 'OccipitalParietal', 'FrontalLeft', 'FrontalRight', 'CentralLeft',
             'CentralRight', 'Appendix']

other_loss_list = ['FrontalLeft', 'FrontalRight', 'CentralRight', 'Appendix']  # areas trained with different loss function
loss_function = 'weighted_dice_loss4 '
training_type = 'ADNIskull_300a'

location = 'local'

if location == 'local':
    test_set = 'Hammers'  # 'Hammers'
    method = 'MV'
if location == 'cluster':
    test_set = sys.argv[1]
    method = sys.argv[2]
show_results = False
text_colour = 'white'
train_path, test_path, lobe_path = path_per_type(location, training_type)

if method == 'MV':
    print('Evaluating concatenation segmentations via majority vote')
if method == 'WB':
    print('Evaluating concatenation segmentations via whole-brain-network')

if location == 'local':
    whole_brain_weight_path = '/home/jara/Savine/datasize176_208_160/hyperparametertuning/AllLobes/Initial_LRE200/' \
                              'ModelDetails0.001/model_weight.h5'
    save_segmentation_path = os.path.join('/home/jara/Savine/datasize176_208_160/results', test_set + '_segmentations')
    if test_set == 'ADNI':
        test_list = np.load('AdditionalFiles/adni_test_set.npy')
        test_path = '/home/jara/Savine/datasize176_208_160/ADNI/TestSkullStripped'
if location == 'cluster':
    whole_brain_weight_path = '/media/data/smartens/data/datasize176_208_160/hyperparametertuning/AllLobes/' \
                              'Initial_LRE200DiceAll/ModelDetails0.001/model_weight.h5'
    save_segmentation_path = os.path.join('/media/data/smartens/results', test_set + '_segmentations')
    if test_set == 'ADNI':
        test_list = np.load('/media/data/smartens/data/datasize176_208_160/ADNI/adni_test_set.npy')
        test_path = '/media/data/smartens/data/datasize176_208_160/ADNI/TestSkullStripped'
initial_lr = 0.001

ref_shape = ref_shape_from_dir(test_path)
print('Reference shape:', ref_shape)

print('Training', training_type)
print('Testing on', test_set)

if not os.path.exists(save_segmentation_path):
    os.makedirs(save_segmentation_path)

# make file to save per class to avoid large array in memory
text_path = "".join(save_segmentation_path + '/DicePerClass' + method +'.csv')
file = open(text_path, 'w+')
file.close()

if test_set == 'Hammers':
    test_list = ['27', '28', '29', '30']

# image size Parameters
params = {'batch_size': num_batch,
          'dim_x': ref_shape[0],
          'dim_y': ref_shape[1],
          'dim_z': ref_shape[2],
          'n_classes': 0,
          'shuffle': True,
          'verbose': verbose_generator}

para_decay_auto = {'initial_lr': initial_lr,
                   'drop_percent': 0.5,
                   'patience': 15,
                   'threshold_epsilon': 0.0001,
                   'momentum': 0.8,
                   'nesterov': True}

DC_FG_avg_all_test = []
# DC_all_classes_all_test = np.empty((len(test_list), 84))

for t, test_idx in enumerate(test_list):
    print('Evaluating', test_idx, '(', t+1, '/', len(test_list), 'scans)')
    # if t == 1:
    #     break
    time_0 = time.time()
    X_i, Y_i = test_loader(test_set, test_path, test_idx)
    Y_true_all_labels = np.asarray(Y_i)

    # reset everything
    old_binary = np.zeros(ref_shape)
    old_labels = np.zeros(ref_shape)
    areas_list = []
    for lobe_i, lobe in enumerate(lobe_list):
        if lobe not in other_loss_list:
            hyper_parameter_to_evaluate = 'Initial_LR'  # using the results of 2 different loss functions
            parameter_save = os.path.join(lobe_path, training_type, lobe, hyper_parameter_to_evaluate +
                                          'E' + str(n_epoch))
        else:
            hyper_parameter_to_evaluate = 'LossFunction'
            parameter_save = os.path.join(lobe_path, training_type, lobe, hyper_parameter_to_evaluate + 'E'
                                          + str(n_epoch), loss_function)
        areas = lobe_selection(lobe)
        n_classes = len(areas) + 1
        for area in areas:
            areas_list.append(area)
        print('\nSelecting areas from', lobe, 'lobe #', lobe_i+1)
        print('Number of classes including background:', n_classes)
        params['n_classes'] = n_classes
        weight_path = os.path.join(parameter_save, 'ModelDetails' + str(initial_lr), 'model_weight.h5')

        # reshaping test volumes and removing irrelevant labels
        Y_true = remove_areas(Y_true_all_labels, areas)
        Y_i = encode1hot(Y_true)
        dimx = np.expand_dims(X_i, axis=0)
        dimy = np.expand_dims(Y_i, axis=0)

        # Construct the model per lobe
        print("Constructing model")
        unet_model = unet(params['dim_x'], params['dim_y'], params['dim_z'], 1, nCommon_multiple, n_classes, alpha_relu)
        unet_model.load_weights(weight_path)
        print("Loading pre-trained model from", weight_path)

        opt = Adam(lr=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        unet_model.compile(optimizer=opt, loss=[dice_loss], metrics=[dice_coef_prob])

        results_i = unet_model.evaluate(dimx, dimy, batch_size=1)
        print('Probabilistic Dice for lobe:', results_i)

        seg_images = unet_model.predict(np.expand_dims(X_i, 0), batch_size=1)
        Y_pred_data = np.squeeze(np.argmax(seg_images, axis=-1))
        Dice_binary, _ = Metrics(Y_true, retrieve_areas(Y_pred_data, areas)).DSC_FG_binary()
        print('Binary Dice (FG) for lobe:', Dice_binary)
        if show_results:
            show_differences(X_i, Y_true, Y_pred_data, results_i[1], text_colour)

        del unet_model  # otherwise it has trouble with the varying number of classes)

        # make all voxels which have labels = 1, for finding overlap
        new_binary = np.where(Y_pred_data > 1, 1, Y_pred_data)
        old_binary = np.where(old_labels > 1, 1, old_labels)
        summation = old_binary + new_binary
        overlap = np.where(summation == 2, 1, np.zeros(summation.shape))

        new_labels = labels_from_onehot(Y_pred_data, areas)  # integers

        if len(np.unique(overlap)) == 1:
            print('No overlap in scan', test_idx, '(assessed', lobe_i+1, '/', len(lobe_list), 'lobes)')
        else:
            print('There is overlap in', test_idx, '(assessed', lobe_i+1, '/', len(lobe_list), 'lobes)')

        if len(np.unique(overlap)) != 1 and lobe_i != 0:
            # max_slice_of_planes_viewer(np.expand_dims(np.squeeze(X_i), 0), np.expand_dims(summation, 0), True)
            n_voxels, location, overlap_volume = show_overlap(X_i, old_binary, new_binary, test_idx, text_colour)
            # plt.savefig(os.path.join(save_overlap_path, 'Overlap' + test_idx + '_' + str(n_voxels) + 'voxels.png'), facecolor="black")
            # print('Saving figure')
            if location == 'local':
                plt.show()

            print('Merging', n_voxels, 'voxels')
            if method == 'MV':
                merged_Y_pred = Overlap(new_labels, old_labels, overlap).majority_vote(cube_size=3)
            if method == 'WB':
                merged_Y_pred = Overlap(new_labels, old_labels, overlap).whole_brain_network(X_i,
                                                                                             whole_brain_weight_path)
            Y_true = remove_areas(Y_true_all_labels, areas_list)
            if any(label > 83.5 for label in np.unique(merged_Y_pred)):
                print('Something is going wrong with the addition')
                break
            foreground_dice, _ = Metrics(Y_true, merged_Y_pred).DSC_FG_binary()
            if lobe_i != len(lobe_list)-1:
                total_dice, _ = Metrics(Y_true, merged_Y_pred).DSC_binary()
            else:
                total_dice, dice_per_area = Metrics(Y_true, merged_Y_pred).DSC_binary()
            print('New FG binary Dice Coefficient after merging:', foreground_dice)
            print('New average binary Dice Coefficient (including BG) after merging:', total_dice)
            old_labels = merged_Y_pred
        else:
            old_labels += new_labels
        print('Continuing to next lobe\n')
        # old_labels = new_labels
    DC_FG_avg_all_test.append(foreground_dice)
    # with open(text_path, mode='w') as file:
    #     file_writer = csv.writer(file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     file_writer.writerow(dice_per_area)
    append_list_as_row(text_path, dice_per_area)
    # DC_all_classes_all_test[t, :] = dice_per_area
    print('Average FG Dice thus far:', np.sum(DC_FG_avg_all_test)/len(DC_FG_avg_all_test))
    if t == 0:
        time_per_scan = time.time() - time_0
        print('Time it takes to segment one scan:', time_per_scan)
    np.save(save_segmentation_path + '/' + method + test_idx, old_labels, allow_pickle=True)


print('Time it takes to segment one scan on average:', (time.time()-time_0)/len(DC_FG_avg_all_test))
print('Average foreground Dice over all test scans:', np.sum(DC_FG_avg_all_test)/len(DC_FG_avg_all_test),
      '(', np.std(DC_FG_avg_all_test), ')')

# avg_class = np.mean(DC_all_classes_all_test, axis=0)
# for l_id in np.arange(84):
#     print(l_id, round(avg_class[l_id], 3))
# print('Average Dice per class', avg_class)
