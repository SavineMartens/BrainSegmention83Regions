import numpy as np
import os
from nilearn import image
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import glob
from Visualisation import *

path = '/home/jara/Savine/ADNI_sample176_208_160'
x_path = glob.glob(path + '/*cropbrain.nii.gz')
y_path = glob.glob(path + '/*cropseg.nii.gz')

x_path.sort()
y_path.sort()

x_data = np.empty((len(x_path), 176, 208, 160))
y_data = np.empty((len(x_path), 176, 208, 160))

for i, subject in enumerate(x_path):
    print(i)
    x_data[i, ] = nib.load(x_path[i]).get_data()
    y_data[i, ] = nib.load(y_path[i]).get_data()

max_slice_of_planes_viewer_mc(x_data, y_data, True)


