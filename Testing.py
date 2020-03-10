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
from sklearn.metrics import confusion_matrix
import seaborn as sns

lobe = 'FrontalRight'
location = 'local'
sample_len = 30

if location == 'cluster':
    training_type = sys.argv[1]  # "Baseline" / "DA_Hammers" / "DA_ADNI" / "BL_ADNI" / 'BL_Hammers' / 'BL_skull'
    test_type = sys.argv[2]  # "ADNI" / 'Hammers'
    text_colour = 'black'
if location == 'local':
    training_type = 'ADNIskull_300a'
    loss_function = 'weighted_dice_loss4 '
    test_type = 'Hammers'
    text_colour = 'white'
it_val = 0.001
hyper_parameter_to_evaluate = 'LossFunction'#'Initial_LR'
train_path, test_path, lobe_path = path_per_type(location, training_type)
if training_type != 'AllLobes':
    print('Training type:', str(training_type), ' with learning rate', str(it_val), 'testing on', test_type)
    if hyper_parameter_to_evaluate == 'Initial_LR':
        parameter_save = os.path.join(lobe_path, training_type, lobe, hyper_parameter_to_evaluate + 'E' + str(n_epoch))
    if hyper_parameter_to_evaluate == 'LossFunction':
        parameter_save = os.path.join(lobe_path, training_type, lobe, hyper_parameter_to_evaluate + 'E'
                                                       + str(n_epoch), loss_function)

    # parameter_save = os.path.join(save_path, training_type, lobe, lateralisation +
    #                               hyper_parameter_to_evaluate + 'E' + str(n_epoch))
    areas = lobe_selection(lobe)
    n_classes = len(areas) + 1
    print('Selecting areas from', lobe, 'lobe')
    print('Number of classes including background:', n_classes)
    print('Areas used in classification:', areas)

elif training_type == 'AllLobes':
    print('Training type:', str(training_type), ' with learning rate', str(it_val))
    parameter_save = os.path.join(lobe_path, training_type, hyper_parameter_to_evaluate + 'E' + str(n_epoch))
    print('Selecting areas from whole brain')
    n_classes = 9  # including background
    print('Number of classes including background:', n_classes)


ref_shape = [176, 208, 160]
# ref_shape = ref_shape_from_dir(test_path)
print('Reference shape:', ref_shape)


model_save_path = os.path.join(parameter_save, 'ModelDetails' + str(it_val))
weight_path = os.path.join(model_save_path, 'model_weight.h5')

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


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

# Construct the model
print("Constructing model")
unet_model = unet(params['dim_x'], params['dim_y'], params['dim_z'], 1, nCommon_multiple, n_classes, alpha_relu)
Model_Summary = unet_model.summary()
unet_model.load_weights(weight_path)
print("Loading pre-trained model from directory:", model_save_path)

opt = Adam(lr=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
unet_model.compile(optimizer=opt, loss=[dice_loss], metrics=[dice_coef_prob])

"""
test
"""

loss_overall = []
DC_all = []
DC_FG_all = []
DC_all_own = []
AVD_all = []
n_missing_labels = []
DC_prop_all = []

if test_type == 'ADNI':
    print("Step 3: testing on control adni")
    if location == 'cluster':
        test_path = '/media/data/smartens/data/datasize176_208_160/ADNI/Test'
        test_list = np.load('/media/data/smartens/data/datasize176_208_160/ADNI/adni_test_set.npy')
        if 'skull' in training_type:
            test_path = '/media/data/smartens/data/datasize176_208_160/ADNI/TestSkullStripped'
    if location == 'local':
        if 'skull' in training_type:
            test_path = '/home/jara/Savine/datasize176_208_160/ADNI/TestSkullStripped'
        else:
            test_path = '/home/jara/Savine/datasize176_208_160/ADNI/Test'
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
        if 'skull' in training_type or training_type == 'AllLobes':
            X_i = nib.load("".join(test_path + '/' + idx + 'cropbrain.nii.gz'))
            Y_i = nib.load("".join(test_path + '/' + idx + 'cropseg.nii.gz'))
        else:
            X_i = nib.load("".join(test_path + '/' + idx + '_blcropbrain.nii.gz'))
            Y_i = nib.load("".join(test_path + '/' + idx + '_blcropseg.nii.gz'))
        X_i = np.asarray(X_i.get_data())
        X_i = np.expand_dims(X_i, axis=4)
        Y_true_all = np.asarray(Y_i.get_data())
        if training_type != 'AllLobes':
            Y_true_sel = remove_areas(Y_true_all, areas)
        else:
            Y_true_sel = areas_to_lobes(Y_true_all)
        Y_i = encode1hot(Y_true_sel)
        dimx = np.expand_dims(X_i, axis=0)
        dimy = np.expand_dims(Y_i, axis=0)
        results_i = unet_model.evaluate(dimx, dimy, batch_size=1)
        print(results_i)

        # make segmented images
        if glob.glob(pred_seg_path + '/worst*.npy'):
            if idx in selection_segmentation:
                seg_images = unet_model.predict(np.expand_dims(X_i, 0), batch_size=1)
                Y_pred_data = np.argmax(seg_images, axis=-1)
                print('Number of classes in prediction:', len(np.unique(Y_pred_data)))
                Y_vals = {'Y_true': Y_true_sel,
                          'Y_pred': labels_from_onehot(np.squeeze(Y_pred_data), areas)}
                AVD_means, AVD_all_i = Metrics(**Y_vals).AVD()
                print('AVD classes:', AVD_all_i)
                show_differences(X_i, Y_i, Y_pred_data, results_i[1], text_colour)
                if idx in best_list:
                    plt.savefig(os.path.join(pred_seg_path, 'DifferenceADNIbest' + idx + 'Dice' +
                                             str(round(results_i[1], 3)) + '.png'), format='png')
                if idx in worst_list:
                    plt.savefig(os.path.join(pred_seg_path, 'DifferenceADNIworst' + idx + 'Dice' +
                                             str(round(results_i[1], 3)) + '.png'), format='png')
        else:
            seg_images = np.squeeze(unet_model.predict(np.expand_dims(X_i, 0), batch_size=1))
            Y_pred_data = np.argmax(seg_images,
                                    axis=-1)  # np.load('/home/jara/Savine/datasize176_208_160/hyperparametertuning/ADNIskull_300a/FrontalLeft/Initial_LRE200/PredictedSegmentationADNIskull_300a0.001/Y_pred_data2020-02-04_15:25:03.npy')#
            Y_pred_sel = retrieve_areas(Y_pred_data, areas)
            Y_pred = encode1hot_missing_labels(Y_pred_sel, areas)
            missing_labels = len(np.unique(Y_pred_sel))
            n_missing_labels.append(n_classes - missing_labels)
            print('labels in segmentation:', np.unique(Y_pred_sel), '(', missing_labels, ')')
            # DC_own, DC_classes, DC_foreground = DiceCoefficientIndividualClass(np.squeeze(Y_i), Y_pred)
            DC_own, DC_classes_i = Metrics(Y_true_sel, Y_pred_sel).DSC()
            DC_foreground, DC_FG_i = Metrics(Y_true_sel, Y_pred_sel).DSC_FG()
            print('Own Dice:', DC_own)
            print('Own FG dice:', DC_foreground)
            print('per class', DC_classes_i)

            # calculate AVD
            Y_vals = {'Y_true': Y_true_sel.copy(),
                      'Y_pred': Y_pred_sel.copy()}
            AVD_mean, AVD_all_i = Metrics(**Y_vals).AVD()
            print('AVD classes:', AVD_all_i)

            # show figure
            APV_params = {'X': np.squeeze(X_i),
                          'Y_true': Y_true_sel.copy(),
                          'Y_pred': Y_pred_sel.copy()}
            AnatomicalPlaneViewer(**APV_params).show_differences(results_i[1], AVD_all_i)
            # plt.savefig(os.path.join(pred_seg_path,
            #                          'DifferenceADNI' + idx + 'Dice' + str(round(results_i[1], 3)) + '.png')
            #             , format='png', facecolor="black")
            plt.show()
            # AnatomicalPlaneViewer(np.squeeze(X_i), Y_true_sel, Y_pred_sel).legend('prediction')
            # plt.show()
            # store results
            AVD_all.append(AVD_mean)
            loss_overall.append(results_i[0])
            DC_all.append(results_i[1])
            DC_FG_all.append(DC_foreground)
            DC_all_own.append(DC_own)


if test_type == 'Hammers':
    print("Step 3: testing on test set Hammers")
    if location == 'cluster':
        test_path = '/media/data/smartens/data/datasize176_208_160/Hammers'
        if 'skull' in training_type:
            test_path = '/media/data/smartens/data/datasize176_208_160/Hammers_skull'
    if location == 'local':
        test_path = '/home/jara/Savine/datasize176_208_160/Hammers'
        if 'skull' in training_type:
            test_path = '/home/jara/Savine/datasize176_208_160/Hammers_skull'
    test_index = [26, 27, 28, 29]
    pred_seg_path = os.path.join(parameter_save, 'PredictedSegmentation' + str(training_type) + str(it_val))
    if not os.path.exists(pred_seg_path):
        os.makedirs(pred_seg_path)
    DC_classes_all = np.empty((len(test_index), n_classes))
    for idx in test_index:
        ID = str(idx + 1)
        ID = ID.zfill(2)
        X_i = nib.load("".join(test_path + '/a' + ID + '.nii.gz'))
        X_i = np.asarray(X_i.get_data())
        X_i = np.expand_dims(X_i, axis=4)
        Y_i = nib.load("".join(test_path + '/a' + ID + '-seg.nii.gz'))
        Y_true_all = np.asarray(Y_i.get_data())
        if training_type != 'AllLobes':
            Y_true_sel = remove_areas(Y_true_all, areas)
        else:
            Y_true_sel = areas_to_lobes(Y_true_all)
        Y_i = encode1hot(Y_true_sel)
        dimx = np.expand_dims(X_i, axis=0)
        dimy = np.expand_dims(Y_i, axis=0)
        # AnatomicalPlaneViewer(np.squeeze(X_i), Y_true_sel, Y_true_sel).max_of_slice('gt', 1)
        # plt.show()
        results_i = unet_model.evaluate(dimx, dimy, batch_size=1)
        # check source code: https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/engine/training.py#L470
        print(results_i)

        seg_images = np.squeeze(unet_model.predict(np.expand_dims(X_i, 0), batch_size=1))
        Y_pred_data = np.argmax(seg_images, axis=-1)# np.load('/home/jara/Savine/datasize176_208_160/hyperparametertuning/ADNIskull_300a/FrontalLeft/Initial_LRE200/PredictedSegmentationADNIskull_300a0.001/Y_pred_data2020-02-04_15:25:03.npy')#
        Y_pred_sel = retrieve_areas(Y_pred_data, areas)
        Y_pred = encode1hot_missing_labels(Y_pred_sel, areas)
        missing_labels = len(np.unique(Y_pred_sel))
        n_missing_labels.append(n_classes-missing_labels)
        print('labels in segmentation:', np.unique(Y_pred_sel), '(', missing_labels, ')')
        # DC_own, DC_classes, DC_foreground = DiceCoefficientIndividualClass(np.squeeze(Y_i), Y_pred)
        Dice_prob = Metrics(Y_i, seg_images).DSC_probability()
        print('Own Dice based on probability:', Dice_prob)
        # plot_cm(Y_true_sel, seg_images)
        DC_own, DC_classes_i = Metrics(Y_true_sel, Y_pred_sel).DSC_binary()
        DC_foreground, DC_FG_i = Metrics(Y_true_sel, Y_pred_sel).DSC_FG_binary()
        print('Own binary Dice:', DC_own)
        print('Own binary FG dice:', DC_foreground)
        print('binary per class', DC_classes_i)
        DC_classes_all[idx-26, :] = DC_classes_i

        # calculate AVD
        Y_vals = {'Y_true': Y_true_sel.copy(),
                  'Y_pred': Y_pred_sel.copy()}
        AVD_mean, AVD_all_i = Metrics(**Y_vals).AVD()
        print('AVD classes:', AVD_all_i)

        # show figure
        APV_params = {'X': np.squeeze(X_i),
                      'Y_true': Y_true_sel.copy(),
                      'Y_pred': Y_pred_sel.copy()}
        AnatomicalPlaneViewer(**APV_params).show_differences(results_i[1], AVD_all_i)
        plt.savefig(os.path.join(pred_seg_path, 'DifferenceHammers' + ID + 'Dice' + str(round(results_i[1], 3)) + '.png')
                    , format='png', facecolor="black")
        plt.show()
        # AnatomicalPlaneViewer(np.squeeze(X_i), Y_true_sel, Y_pred_sel).legend('prediction')
        # plt.show()
        # store results
        AVD_all.append(AVD_mean)
        loss_overall.append(results_i[0])
        DC_all.append(results_i[1])
        DC_FG_all.append(DC_foreground)
        DC_all_own.append(DC_own)
        DC_prop_all.append(Dice_prob)


results = [np.mean(DC_all), sum(loss_overall)/len(loss_overall)]
print('Selected scans for testing from:', test_path)
print('Dice coefficient test data:', results[0])
print('Dice coefficient standard deviation', np.std(DC_all))
print('Absolute Volume Difference:', np.mean(AVD_all), '(', np.std(AVD_all), ')' )
print('Missing classes:', n_missing_labels)
print('Averaged binary Dice per class', np.round(np.mean(DC_classes_all, axis=0), 3))
print('Own Dice based on probability all:', np.mean(DC_prop_all), '(', np.std(DC_prop_all), ')')
print('Own Dice binary all:', np.mean(DC_all_own), '(', np.std(DC_all_own), ')')
print('Own FG all:', np.mean(DC_FG_all), '(', np.std(DC_FG_all), ')')

np.save(os.path.join(pred_seg_path, 'DiceCoefficientResultsfor' + test_type + '.npy'), DC_all, allow_pickle=True)
np.save(os.path.join(pred_seg_path, 'AbsoluteVolumeDifferenceResultsfor' + test_type + '.npy'), AVD_all, allow_pickle=True)

# if test_type == 'ADNI':
#     n=5
#     list2 = test_list
#     list1 = DC_all
#
#     list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
#     best = list2[-n - 1:-1]
#     worst = list2[0:n]
#
#     np.save(os.path.join(pred_seg_path, 'worst' + test_type + '.npy'), worst, allow_pickle=True)
#     np.save(os.path.join(pred_seg_path, 'best' + test_type + '.npy'), best, allow_pickle=True)



