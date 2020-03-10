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
######################################################################################################################
sample_len = 30
# lobe_list = ['Temporal', 'Parietal', 'Frontal', 'Central', 'Appendix']
lobe = 'Frontal'
location = "cluster"        # local / cluster
# n_epoch = 1
lateralisation = 'Left'

if location == 'cluster':
    training_type = sys.argv[1]  # "Baseline" / "DA_Hammers" / "DA_ADNI"
    it_val = float(sys.argv[2])
elif location == 'local':
    training_type = "ADNIskull_200"  # "ADNIskull_200" "Baseline" / "DA_Hammers" / "DA_ADNI" / "BL_ADNI" / 'BL_Hammers' / 'BL_skull' /
    # "BL_ADNI_wo_test" / ADNI_100 / ADNI_200 / ADNI_400 / ADNI_600
    it_val = 0.001

train_path, test_path, save_path = path_per_type(location, training_type)
ref_shape = ref_shape_from_dir(test_path)
print('Reference shape:', ref_shape)

print('Training', training_type)

hyper_parameter_to_evaluate = 'Initial_LR'  # 'Batch_Size' / 'Alpha_ReLu' / 'TrainingSize'
parameter_save = os.path.join(save_path, training_type, lobe, lateralisation +
                          hyper_parameter_to_evaluate + 'E' + str(n_epoch))
if not os.path.exists(parameter_save):
    os.makedirs(parameter_save)

if hyper_parameter_to_evaluate == 'Initial_LR':
    initial_lr = it_val
if hyper_parameter_to_evaluate == 'Batch_Size':
    num_batch = it_val
if hyper_parameter_to_evaluate == 'Alpha_ReLu':
    alpha_relu = it_val

print('Evaluation of hyper parameter:', hyper_parameter_to_evaluate, ', value:', it_val)

model_save_path = os.path.join(parameter_save, 'ModelDetails' + str(it_val))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
check_path = os.path.join(model_save_path, 'weights.{epoch:03d}-{val_loss:.2f}.hdf5')
model_path = os.path.join(model_save_path, 'Vkeras_A1_model.hdf5')
json_path = os.path.join(model_save_path, 'model_architecture.json')
weight_path = os.path.join(model_save_path, 'model_weight.h5')
train_history_path = os.path.join(model_save_path, "train_history.csv")   # change name

areas = area_selection(lobe, lateralisation)
n_classes = len(areas) + 1
print('Selecting areas from', lobe, 'lobe of', lateralisation, 'side(s)')
print('Number of classes including background:', n_classes)
print('Areas used in classification:', areas)
# outchannel = 2
# G = 2  # the number of GPUs

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

# image size Parameters
params = {'batch_size': num_batch,
          'dim_x': ref_shape[0],
          'dim_y': ref_shape[1],
          'dim_z': ref_shape[2],
          'n_classes': n_classes,
          'shuffle': True,
          'verbose': verbose_generator}

# Generators
if training_type == 'DA_Hammers':
    list_training = np.load(os.path.join(train_path, 'ListTraining.npy'))
    steps_per_epoch = len(list_training)
    training_generator = DataGenerator(**params).generate_training(list_training, train_path, areas,
                                                                   training_size=steps_per_epoch)
if training_type == 'DA_ADNI':
    adni_list = np.load(os.path.join(train_path, 'adni_list_full_labels.npy'))
    steps_per_epoch = len(adni_list)
    training_generator = DataGenerator(**params).adni_training(adni_list, train_path, areas,
                                                                   training_size=steps_per_epoch)
if training_type == 'Baseline' or training_type == 'BL_skull':
    list_training = train_idx
    steps_per_epoch = len(train_idx)
    training_generator = DataGenerator(**params).generate_validation(list_training, areas, test_path)

if training_type == 'BL_ADNI':
    list_training = train_idx
    if location == 'cluster':
        adni_list = np.load(os.path.join(train_path, 'adni_list_full_labels.npy'))
    elif location == 'local':
        adni_list = np.load(os.path.join(train_path, 'adni_list.npy'))
    list_training = np.append(list_training, adni_list)
    # print(list_training)
    steps_per_epoch = len(train_idx) + len(adni_list)
    training_generator = DataGenerator(**params).mixed_training(list_training, train_path, test_path, areas,
                                                                   training_size=steps_per_epoch)

if training_type == 'BL_ADNI_wo_test':
    list_training = train_idx
    if location == 'cluster':
        adni_list = np.load(os.path.join(train_path, 'adni_train_set.npy'))
    elif location == 'local':
        adni_list = np.load(os.path.join(train_path, 'adni_list.npy'))
    list_training = np.append(list_training, adni_list)
    # print(list_training)
    steps_per_epoch = len(train_idx) + len(adni_list)
    training_generator = DataGenerator(**params).mixed_training(list_training, train_path, test_path, areas,
                                                                   training_size=steps_per_epoch)

if training_type == 'BL_Hammers':
    list_training = train_idx
    list_hammers = np.load(os.path.join(train_path, 'ListTraining10.npy'))
    list_training = np.append(list_training, list_hammers)
    # print(list_training)
    steps_per_epoch = len(train_idx) + len(list_hammers)
    print('Number of training samples:', steps_per_epoch)
    training_generator = DataGenerator(**params).mixed_training(list_training, train_path, test_path, areas,
                                                                   training_size=steps_per_epoch)

if 'ADNI' in training_type and 'DA' not in training_type and 'BL' not in training_type:
    # uses BL + selection of ADNI
    list_training = train_idx
    steps_per_epoch = ''.join([n for n in training_type if n.isdigit()])
    if training_type[-1].isdigit():
        adni_list = np.load(os.path.join(train_path, 'adni_train' + steps_per_epoch + '_set.npy'))
    else:
        version = str(steps_per_epoch) + training_type[-1]
        adni_list = np.load(os.path.join(train_path, 'adni_train' + version + '_set.npy'))
    list_training = np.append(list_training, adni_list)
    steps_per_epoch = len(train_idx) + len(adni_list)
    training_generator = DataGenerator(**params).mixed_training(list_training, train_path, test_path, areas,
                                                                training_size=10)
validation_generator = DataGenerator(**params).generate_validation(val_idx, areas, test_path)

# Construct the model
print("Constructing model")
unet_model = unet(params['dim_x'], params['dim_y'], params['dim_z'], 1, nCommon_multiple, n_classes, alpha_relu)
Model_Summary = unet_model.summary()

if resume_pretrained:
    unet_model.load_weights(weight_path)
    print("Loading pre-trained model from disk!")
else:
    print('Training a new model!')


"""
train
"""
opt = Adam(lr=para_decay_auto['initial_lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
unet_model.compile(optimizer=opt, loss=[dice_loss], metrics=[dice_coef_prob])
# Train model on dataset
print("Step 1: load images for training and validation")

auto_decay = ReduceLROnPlateau(monitor='val_loss',
                               factor=para_decay_auto['drop_percent'],
                               patience=para_decay_auto['patience'],
                               verbose=1,
                               mode='min',
                               min_delta=para_decay_auto['threshold_epsilon'],
                               cooldown=0,
                               min_lr=0)

class LossHistory_auto(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(auto_decay)
        print('lr: ', K.eval(self.model.optimizer.lr))

loss_history = LossHistory_auto()

# to check whether the validation performance is improved, the learning rate will decay if not
checkpoint = ModelCheckpoint(check_path,
                             monitor='val_dice_coef_prob',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max',
                             period=1)
checkpoint_eachEpoch = ModelCheckpoint(weight_path,
                                       monitor='val_dice_coef_prob',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='max',
                                       period=1)
csv_logger = CSVLogger(train_history_path, separator=',', append=True)

EarlyStoppingEpochs = EarlyStopping(monitor='val_dice_coef_prob', min_delta=0, patience=30, verbose=1, mode='max',
                                    baseline=None, restore_best_weights=True)

callbacks_list = [checkpoint, checkpoint_eachEpoch, loss_history, csv_logger, EarlyStoppingEpochs]  # auto_decay,

history = unet_model.fit_generator(generator=training_generator,
                                   steps_per_epoch=steps_per_epoch, epochs=n_epoch,
                                   verbose=2, callbacks=callbacks_list,
                                   validation_data=validation_generator,
                                   validation_steps=len(val_idx)//params['batch_size'], initial_epoch=0)

if save_this_weight:
    print("Saving model")
    # save keras model
    unet_model.save(model_path)
    print("Saving architecture and weight")
    model_json = unet_model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    unet_model.save_weights(weight_path)
    print("Saved!")

"""
test
"""
print("Step 3: testing on remaining", len(test_index))

loss_overall = []
DC_all = []

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
    print('Number of classes in prediction', len(np.unique(Y_pred_data)))

    pred_seg_path = os.path.join(parameter_save, 'PredictedSegmentation' + str(training_type) + str(it_val))
    if not os.path.exists(pred_seg_path):
        os.makedirs(pred_seg_path)
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_stamp = time_stamp.replace(" ", "_")
    np.save(os.path.join(pred_seg_path, 'Y_pred_data' + time_stamp), Y_pred_data, allow_pickle=True)
    loss_overall.append(results_i[0])
    DC_all.append(results_i[1])


results = [np.mean(DC_all), sum(loss_overall)/len(loss_overall)]
print('Dice coefficient test data:', results[0])
acc = history.history['dice_coef']
val_acc = history.history['val_dice_coef']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', color='red', label='Training')
plt.plot(epochs, val_acc, 'b', color='blue', label='Validation')
plt.hlines(results[0], epochs[0], epochs[-1], label='Test')
plt.title('Dice coefficient, C' + str(nCommon_multiple) + ' T' + str(num_test) + ' V'
          + str(num_val) + ' B' + str(num_batch) + ' ' + str(hyper_parameter_to_evaluate) + str(it_val))
plt.legend()
plt.ylim(0, 1)
time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
time_stamp = time_stamp.replace(" ", "_")
result_str = "{:.2f}".format(results[1], 2)
train_stamp = "".join('DC' + result_str + 'C' + str(nCommon_multiple) +
                      'T' + str(num_test) + 'V' + str(num_val) + 'E' + str(n_epoch) + '_' +
                      str(hyper_parameter_to_evaluate) + str(it_val))
save_str = "".join(time_stamp + train_stamp + '.png')
if n_epoch > 1:
    plt.savefig(os.path.join(parameter_save, save_str))
if location == 'local':
    plt.show()

np.save(os.path.join(parameter_save, 'results' + time_stamp + str(hyper_parameter_to_evaluate) + str(it_val)), results)
