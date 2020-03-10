import numpy as np
import nibabel as nib
from Shortlist import *
from Visualisation2 import *
lobe = 'FrontalLeft'
test_path = '/home/jara/Savine/datasize176_208_160/Hammers_skull'
areas = lobe_selection(lobe)
ID = str(27)
X_i = nib.load("".join(test_path + '/a' + ID + '.nii.gz'))
X_i = np.asarray(X_i.get_data())
Y_i = nib.load("".join(test_path + '/a' + ID + '-seg.nii.gz'))
Y_true_all = np.asarray(Y_i.get_data())
Y_true = remove_areas(Y_true_all, areas)

AnatomicalPlaneViewer(np.squeeze(X_i), Y_true, Y_true).legend2('gt')
# AnatomicalPlaneViewer(np.squeeze(X_i), Y_true, Y_true).legend('gt')
