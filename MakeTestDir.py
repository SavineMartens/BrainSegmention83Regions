import sys
import numpy as np
import os
import nibabel as nib

test_list = np.load('/media/data/smartens/data/datasize176_208_160/ADNI/adni_train200_set.npy')
test_path = '/media/data/smartens/data/datasize176_208_160/ADNI'

save_path = '/media/data/smartens/data/datasize176_208_160/ADNI/ADNI_200'
# make test directory
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
for i, idx in enumerate(test_list):
    print(i)
    X_i = nib.load("".join(test_path + '/' + idx + 'cropbrain.nii.gz'))
    Y_i = nib.load("".join(test_path + '/' + idx + 'cropseg.nii.gz'))
    nib.save(X_i, "".join(save_path + '/' + idx + 'cropbrain.nii.gz'))
    nib.save(Y_i, "".join(save_path + '/' + idx + 'cropseg.nii.gz'))
