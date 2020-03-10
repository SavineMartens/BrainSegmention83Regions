import numpy as np
import keras
import os
import nibabel as nib
from Shortlist import *
import random
import glob

location = 'cluster'
if location == 'cluster':
    path = '/media/data/smartens/data/datasize176_208_160/ADNI'
elif location == 'local':
    path = '/home/jara/Savine/ADNI_sample176_208_160'

list = glob.glob(path + '/*cropseg.nii.gz')

adni_list = []

for i, name in enumerate(list):
    Y_i = nib.load(list[i])
    Y_i = np.asarray(Y_i.get_data())
    print('length of Y:', len(np.unique(Y_i)))
    if len(np.unique(Y_i)) == 84:
        if location == 'cluster':
            step1 = name.replace('/media/data/smartens/data/datasize176_208_160/ADNI/', '')
        elif location == 'local':
            step1 = name.replace('/home/jara/Savine/ADNI_sample176_208_160/', '')
        step2 = step1.replace('cropseg.nii.gz', '')
        adni_list.append(step2)
        print(i, step2)

np.asarray(adni_list)

np.save(os.path.join(path, 'adni_list_full_labels'), adni_list, allow_pickle=True)
