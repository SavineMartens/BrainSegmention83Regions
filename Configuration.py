import numpy as np
import os
import sys
from itertools import groupby
"""""
Configuration file for main in which 3D segmentation is performed on brain MRI scans
"""""

# training
# lobe = 'Appendix'  # 'Temporal' / 'Parietal' / 'Frontal' / 'Central' / 'Appendix'
# lateralisation = 'Left'  # 'Both' / 'Left' / 'Right'

num_test = 4  # change this to number of subjects you want to test on
num_val = 4  # number subjects to use for validation
n_epoch = 200
nCommon_multiple = 8
verbose_generator = False

resume_pretrained = False  # load previously trained model for future refinement
save_this_weight = True  # if just test, choose False, keep pre-trained weight

# optimizer
initial_lr = 0.001  # outcome after tuning with fixed learning rate
num_batch = 1
# steps_per_epoch = 50  # in case of data augmentation (DA)
alpha_relu = 0.2


# directories
def path_per_type(location, training_type):
    if location == "local":
        test_path = '/home/jara/Savine/datasize176_208_160/Hammers'
        save_path = '/home/jara/Savine/datasize176_208_160/hyperparametertuning'
        train_path = []
        if training_type == 'DA_Hammers' or training_type == 'BL_Hammers':
            print('No augmented Hammers data on this computer')
            train_path = []
        elif 'skull' in training_type and 'ADNI' in training_type:
            train_path = os.path.join('/home/jara/Savine/datasize176_208_160/ADNI', training_type)
            test_path = '/home/jara/Savine/datasize176_208_160/Hammers_skull'
        elif 'ADNI' in training_type:
            train_path = '/home/jara/Savine/datasize176_208_160/ADNI/ADNI_200'
        elif training_type == 'Baseline':
            train_path = test_path
        elif training_type == 'BL_skull':
            train_path = '/home/jara/Savine/HammersSkullStripped176_224_240'
            test_path = train_path
        elif training_type == 'AllLobes':
            train_path = '/home/jara/Savine/datasize176_208_160/ADNI/ADNIskull_200'
            test_path = '/home/jara/Savine/datasize176_208_160/Hammers_skull'
        elif training_type == 'TestSkullStripped':
            train_path = '/home/jara/Savine/datasize176_208_160/ADNI/TestSkullStripped'
            test_path = '/home/jara/Savine/datasize176_208_160/Hammers_skull'
    else:
        test_path = '/media/data/smartens/data/datasize176_208_160/Hammers'
        save_path = '/media/data/smartens/data/datasize176_208_160/hyperparametertuning'
        train_path = []
        # ref_shape = [176, 208, 160]  # np.load(os.path.join(test_path, 'ref_shape.npy'))
        if training_type == 'DA_Hammers' or training_type == 'BL_Hammers':
            train_path = '/media/data/smartens/data/datasize176_208_160/HammersAugmented'
        elif 'skull' in training_type and 'ADNI' in training_type:
            if training_type[-1].isdigit():
                train_path = os.path.join('/media/data/smartens/data/datasize176_208_160/ADNI', training_type)
            else:
                train_path = os.path.join('/media/data/smartens/data/datasize176_208_160/ADNI', training_type[0:-1])
            test_path = '/media/data/smartens/data/datasize176_208_160/Hammers_skull'
        elif 'ADNI' in training_type:
            train_path = '/media/data/smartens/data/datasize176_208_160/ADNI'
        elif training_type == 'Baseline':
            train_path = test_path
        elif training_type == 'BL_skull':
            test_path = '/media/data/smartens/data/datasize176_208_160/Hammers_skull'
            # save_path = '/media/data/smartens/data/datasize176_208_160/hyperparametertuning'
            train_path = test_path
            # ref_shape = [160, 208, 160]

    return train_path, test_path, save_path


def ref_shape_from_dir(dir):
    ref_shape = [int(''.join(i)) for is_digit, i in groupby(dir, str.isdigit) if is_digit]

    return ref_shape


