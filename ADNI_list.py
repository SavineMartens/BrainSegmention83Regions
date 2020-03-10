import numpy as np
import os
from nilearn import image
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import glob
from Visualisation import *

location = 'cluster'
if location == 'cluster':
    path = '/media/data/smartens/data/datasize176_208_160/ADNI'
elif location == 'local':
    path = '/home/jara/Savine/ADNI_sample176_208_160'
x_path = glob.glob(path + '/*cropbrain.nii.gz')

adni_list = []

for i, name in enumerate(x_path):
    if location == 'cluster':
        step1 = name.replace('/media/data/smartens/data/datasize176_208_160/ADNI/', '')
    elif location == 'local':
        step1 = name.replace('/home/jara/Savine/ADNI_sample176_208_160/', '')
    step2 = step1.replace('cropbrain.nii.gz', '')
    adni_list.append(step2)
    print(i, step2)

np.asarray(adni_list)

np.save(os.path.join(path, 'adni_list'), adni_list, allow_pickle=True)

################################################################

# Graded training selection
import random
import numpy as np

train_list = np.load('/AdditionalFiles/adni_train_set.npy')

num_train = len(train_list)
id_list = list(range(num_train))

# random100 = random.sample(id_list, 100)
# random200 = random.sample(id_list, 200)
random300 = random.sample(id_list, 300)
# random400 = random.sample(id_list, 400)
# random600 = random.sample(id_list, 600)


# train_list100 = train_list[random100]
# train_list200 = train_list[random200]
train_list300 = train_list[random300]
# train_list400 = train_list[random400]
# train_list600 = train_list[random600]

# np.save('/home/jara/PycharmProjects/GPUcluster/adni_train100_set.npy', train_list100)
# np.save('/home/jara/PycharmProjects/GPUcluster/adni_train200_set.npy', train_list200)
np.save('/AdditionalFiles/adni_train300_set.npy', train_list300, allow_pickle=True)
# np.save('/home/jara/PycharmProjects/GPUcluster/adni_train400_set.npy', train_list400)
# np.save('/home/jara/PycharmProjects/GPUcluster/adni_train600_set.npy', train_list600)


#############################################################################################
# Graded training selection
import random
import numpy as np

train_list = np.load('/AdditionalFiles/adni_train_set.npy')

num_train = len(train_list)
id_list = list(range(num_train))

random300a = random.sample(id_list, 300)
random300b = random.sample(id_list, 300)
random300c = random.sample(id_list, 300)
random300d = random.sample(id_list, 300)
random300e = random.sample(id_list, 300)

train_list300a = train_list[random300a]
train_list300b = train_list[random300b]
train_list300c = train_list[random300c]
train_list300d = train_list[random300d]
train_list300e = train_list[random300e]

list_random_300 = random300a + random300b + random300c + random300d + random300e
unique_list = np.unique(list_random_300)
list_all_300_collected = train_list[unique_list]


np.save('/AdditionalFiles/adni_train300a_set.npy', train_list300a, allow_pickle=True)
np.save('/AdditionalFiles/adni_train300b_set.npy', train_list300b, allow_pickle=True)
np.save('/AdditionalFiles/adni_train300c_set.npy', train_list300c, allow_pickle=True)
np.save('/AdditionalFiles/adni_train300d_set.npy', train_list300d, allow_pickle=True)
np.save('/AdditionalFiles/adni_train300e_set.npy', train_list300e, allow_pickle=True)

np.save('/AdditionalFiles/adni_train300_total_set.npy', list_all_300_collected, allow_pickle=True)