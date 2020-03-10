import numpy as np
import os
from nilearn import image
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

original_dir = '/media/data/smartens/data/ADNI_rotated'  # '/home/jara/Savine/ADNI_rotated'
# str_mask = 'brain_mask.nii.gz'
num_area = 83
adni_list = np.load(os.path.join(original_dir, 'adni_list.npy'))
sample_len = len(adni_list)  # total number of subjects

adni_list = adni_list[:-2]

images = []
targets = []
list_mask = []
feat_path_list = []


def pad(array, reference_shape):
    """
    array: Array to be padded
    ref_shape: tuple of size of ndarray to create
    """
    result = np.zeros(reference_shape)  # Create an array of zeros with the reference shape
    diff = np.asarray(array.shape) - reference_shape
    offsets = abs(diff//2)
    result[offsets[0]:array.shape[0] + offsets[0], offsets[1]:array.shape[1] + offsets[1], offsets[2]:array.shape[2]
           + offsets[2]] = array
    return result


########################################################################################################################
#   BOUNDING BOX
path = original_dir

max_x_width = 0
max_y_width = 0
max_z_width = 0

# Parameters and Initialization
history = {}

# Find common bounding box of all subjects in a cohort
for j, subject in enumerate(adni_list):
    try:
        mask_path = os.path.join(path, subject, 'brain_mask', 'result.nii.gz')
        print(mask_path)
        mask = image.load_img(mask_path).get_data()

        # list the slices which are not empty
        xslice = []
        yslice = []
        zslice = []

        for x in range(mask.shape[0]):
            slice = mask[x, :, :]
            if slice.max() > 0:
                xslice.append(x)
        xslice.sort()

        for y in range(mask.shape[1]):
            slice = mask[:, y, :]
            if slice.max() > 0:
                yslice.append(y)
        yslice.sort()

        for z in range(mask.shape[2]):
            slice = mask[:, :, z]
            if slice.max() > 0:
                zslice.append(z)
        zslice.sort()

        # Error messages
        if 0 == len(xslice) or 0 == len(yslice):
            print('Warning: Please check your dataset, xslice or yslice is 0.')
            break
            # exit()
        if 0 == len(zslice):
            print('no value in mask!')
            break

        # find start and end
        xstart = min(xslice)
        xend = max(xslice)
        ystart = min(yslice)
        yend = max(yslice)
        zstart = min(zslice)
        zend = max(zslice)

        # record the max and min width in a cohort
        x_width = xend - xstart + 1
        y_width = yend - ystart + 1
        z_width = zend - zstart + 1

        if x_width > max_x_width:
            max_x_width = x_width
        if y_width > max_y_width:
            max_y_width = y_width
        if z_width > max_z_width:
            max_z_width = z_width

        print('No.', '%03d:' % j, 'x start from', '%03d,' % xstart, 'end by', '%03d,' % xend, 'width is',
              '%03d.' % x_width)
        print('        ', 'y start from', '%03d,' % ystart, 'end by', '%03d,' % yend, 'height is', '%03d.' % y_width)
        print('        ', 'z start from', '%03d,' % zstart, 'end by', '%03d,' % zend, 'depth is', '%03d.' % z_width)

        bbox = {'x_width': x_width, 'y_width': y_width, 'z_width': z_width, 'xstart': xstart, 'xend': xend,
                'ystart': ystart, 'yend': yend, 'zstart': zstart, 'zend': zend}
        history[subject] = bbox
    except ValueError:
        print(subject, 'not present')
        continue


history['max'] = {'max_x_width': max_x_width, 'max_y_width': max_y_width,
                         'max_z_width': max_z_width}

print('max_x_width: ', max_x_width, 'max_y_width: ', max_y_width,
      'max_z_width: ', max_z_width)

# Step 2 calculate the border among all subjects
nDownSample = 16

x_width_crop = int(np.ceil(max_x_width/nDownSample)*nDownSample)
y_width_crop = int(np.ceil(max_y_width/nDownSample)*nDownSample)
z_width_crop = int(np.ceil(max_z_width/nDownSample)*nDownSample)

###############################################################################################################
# Step 3 crop the data

ref_shape = [x_width_crop, y_width_crop, z_width_crop]
print('Reference cropped shape: ', ref_shape)

save_path = '/media/data/smartens/data/datasize176_208_160/ADNI'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def norm(array):
    """"
    makes mean of array zero and std of 1
    """
    array -= np.mean(array)
    array /= np.std(array)
    return array


for i, subject in enumerate(adni_list):
    # if i == 1560:
    #     continue
    try:
        # loading each image with label and lobe mask
        ID = adni_list[i]
        print(i, ID)
        features = nib.load(os.path.join(path, subject, 'a01', 'result.0.nii.gz')).get_data()
        # print(np.unique(features))
        gt_data = nib.load(os.path.join(path, subject, 'seg', 'result.nii.gz')).get_data()

        # load the start and the end in each bbox
        key = ID
        xstart = history[key]['xstart']
        xend = history[key]['xend']
        ystart = history[key]['ystart']
        yend = history[key]['yend']
        zstart = history[key]['zstart']
        zend = history[key]['zend']

        # filling in cropped volume into reference shape
        crop_data = features[xstart:xend + 1, ystart:yend + 1, zstart:zend + 1]
        crop_gt = gt_data[xstart:xend + 1, ystart:yend + 1, zstart:zend + 1]

        # padding
        crop_data = pad(crop_data, ref_shape)
        crop_gt = pad(crop_gt, ref_shape)

        # normalization
        crop_data = norm(crop_data)

        # saving data
        img = nib.Nifti1Image(crop_data, np.eye(4, 4))
        nib.save(img, os.path.join(save_path, ID + 'cropbrain.nii.gz'))

        img2 = nib.Nifti1Image(crop_gt, np.eye(4, 4))
        nib.save(img2, os.path.join(save_path, ID + 'cropseg.nii.gz'))
    except ValueError:
        print(subject, 'not present')
        continue
