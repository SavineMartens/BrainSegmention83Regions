import numpy as np
import random
import nibabel as nib
from csv import writer


def retrieve_idx(X_test, X_data, verbose):
    test_indices = []
    num_test = X_test.shape[0]
    sample_len = X_data.shape[0]
    ref_shape = [X_data.shape[1], X_data.shape[2], X_data.shape[3]]
    for j in np.arange(num_test):
        com_vol = X_test[j, ]
        for i in np.arange(sample_len):
            ref_vol = X_data[i, ]
            comparison = com_vol == ref_vol
            x = comparison.reshape(1, -1)
            x = x.astype('uint8')
            if x.sum() == ref_shape[0]*ref_shape[1]*ref_shape[2]:
                test_indices.append(i)
                if verbose == 1:
                    print('Selected image ', j, 'of set is index: ', i+1, '/', sample_len)

    return test_indices


def encode1hot(Y):
    '''
    labels shape: [(index,) x, y, z ]
    output shape: [(index,) x, y, z , num_classes]
    '''
    all_labels = np.unique(Y)
    # print('All labels for 1-hot:', all_labels)
    num_classes = len(all_labels)
    # print('length:', num_classes)
    tuple = Y.shape + (num_classes,)
    Y_cat = np.empty(tuple)
    if Y.ndim == 3:
        for i, label in enumerate(all_labels):
            # print(i, label)
            temp = Y == label
            temp = temp.astype('uint8')
            Y_cat[:, :, :, i] = temp
            del temp
    if Y.ndim == 4:
        for i, label in enumerate(all_labels):
            temp = Y == label
            temp = temp.astype('uint8')
            Y_cat[:, :, :, :, i] = temp
            del temp
    return Y_cat


def labels_from_onehot(Y, labels):
    '''
    :param Y: one-hot-encoded segmentation or from 0 to number of areas
    :param labels: numbers between 0 and 83
    :return: labels in integers
    '''
    while Y.ndim > 4:
        Y = np.squeeze(Y)
    # make Y from 0 to number of labels
    if len(np.unique(Y)) == 2:
        Y = np.argmax(Y, axis=-1)
    labels = np.sort(labels)
    Y_empty = np.empty(Y.shape)
    Y_empty = np.where(Y == 0, 0, Y_empty)
    for i, label in enumerate(labels):
        Y_empty = np.where(Y == i+1, int(label), Y_empty)

    return Y_empty

def encode1hot_missing_labels(labels, areas):
    '''
    labels shape: [(index,) x, y, z ]
    output shape: [(index,) x, y, z , num_classes]
    '''
    all_labels = areas
    # labels_in_segmentation = np.unique(labels)
    # print('All labels for 1-hot:', all_labels)
    num_classes = len(all_labels)
    # print('length:', num_classes)
    if labels.ndim == 4:
        tuple = labels.shape + (num_classes,)
        Y_cat = np.empty(tuple)
        for i, label in enumerate(all_labels):
            temp = labels == label
            temp = temp.astype('uint8')
            Y_cat[:, :, :, :, i] = temp
            del temp

    if labels.ndim == 3:
        tuple = labels.shape + (num_classes,)
        Y_cat = np.empty(tuple)
        for i, label in enumerate(all_labels):
            temp = labels == label
            temp = temp.astype('uint8')
            Y_cat[:, :, :, i] = temp
            del temp

    return Y_cat


def DiceCoefficientIndividualClass(Y_true, Y_pred):
    '''''
    input one-hot-encoded! [x, y, z, labels] 
    '''''
    num_classes = Y_true.shape[-1]
    DC_classes = []
    n_pred_classes = Y_pred.shape[-1]
    for c in np.arange(num_classes):
        sel_class_true = Y_true[:, :, :, c]
        if c == n_pred_classes and num_classes != n_pred_classes:
            DC_classes.append(0.0)
            print('Prediction and ground truth don\'t have same number of classes')
            break
        sel_class_pred = Y_pred[:, :, :, c]
        numerator = np.sum(sel_class_true * sel_class_pred)
        denominator = np.sum(sel_class_true + sel_class_pred)
        DC = (2 * numerator)/denominator
        DC_classes.append(DC)
        print('Dice coefficient class', c, ':', DC)

    DC_all = sum(DC_classes) / num_classes
    DC_foreground = sum(DC_classes[1:-1]) / (num_classes-1)
    print('Dice coefficient averaged over all classes:', DC_all)
    print('Dice coefficient average without background', DC_foreground)

    return DC_all, DC_classes, DC_foreground


def Dice_FG(Y_true, Y_pred):
    '''''
    input one-hot-encoded except if Y_pred has less classes! [x, y, z, labels] 
    '''''
    unequal_classes = False
    if len(np.unique(Y_true)) != len(np.unique(Y_pred)):
        unequal_classes = True
        true_labels = np.unique(Y_true)
        pred_labels = np.unique(Y_pred)
        indices = [index for index, element in enumerate(true_labels) if element not in pred_labels]
    # if not one-hot-encoded, make it so
    if len(np.unique(Y_true)) != 2:
        Y_true = encode1hot(Y_true)
    if len(np.unique(Y_pred)) != 2:
        Y_pred = encode1hot(Y_pred)
    num_classes = Y_true.shape[-1]
    if unequal_classes:
        new_pred = np.zeros(Y_true.shape)
        n = 0
        for m in np.arange(num_classes):
            if m not in indices:
                new_pred[:, :, :, m] = Y_pred[:, :, :, n]
                n = n + 1
            if m in indices:
                continue
            if m == len(true_labels)-1:
                Y_pred = new_pred
    DC_classes = []
    for c in np.arange(num_classes):
        sel_class_true = Y_true[:, :, :, c]
        sel_class_pred = Y_pred[:, :, :, c]
        numerator = np.sum(sel_class_true * sel_class_pred)
        denominator = np.sum(sel_class_true + sel_class_pred)
        DC = (2 * numerator)/denominator
        DC_classes.append(DC)
    DC_foreground = sum(DC_classes[1:-1]) / len(DC_classes[1:-1])

    return DC_foreground

def Dice_All(Y_true, Y_pred):
    '''''
    input one-hot-encoded! [x, y, z, labels] 
    '''''
    unequal_classes = False
    if len(np.unique(Y_true)) != len(np.unique(Y_pred)):
        unequal_classes = True
        true_labels = np.unique(Y_true)
        pred_labels = np.unique(Y_pred)
        # missing_labels = set(true_labels) - set(pred_labels)
        indices = [index for index, element in enumerate(true_labels) if element not in pred_labels]
    # if not one-hot-encoded, make it so
    if len(np.unique(Y_true)) != 2:
        Y_true = encode1hot(Y_true)
    if len(np.unique(Y_pred)) != 2:
        Y_pred = encode1hot(Y_pred)
    num_classes = Y_true.shape[-1]
    if unequal_classes:
        new_pred = np.zeros(Y_true.shape)
        n = 0
        for m in np.arange(num_classes):
            if m not in indices:
                new_pred[:, :, :, m] = Y_pred[:, :, :, n]
                n = n + 1
            if m in indices:
                continue
            if m == len(true_labels)-1:
                Y_pred = new_pred
    DC_classes = []
    for c in np.arange(num_classes):
        sel_class_true = Y_true[:, :, :, c]
        sel_class_pred = Y_pred[:, :, :, c]
        numerator = np.sum(sel_class_true * sel_class_pred)
        denominator = np.sum(sel_class_true + sel_class_pred)
        DC = (2 * numerator)/denominator
        DC_classes.append(DC)
    DC_all = sum(DC_classes) / len(DC_classes)

    return DC_all

# used the areas of lobemap.txt, but these are different from kwabmapdescription.txt for the following fuctions


def area_selection(lobe, lateralisation):
    if lobe == 'Occipital':
        areas = [22, 23, 64, 65, 66, 67]
    elif lobe == 'Temporal':
        areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 30, 31, 47, 48, 82, 83]
    elif lobe == ' Parietal':
        areas = [32, 33, 60, 61, 62, 63]
    elif lobe == 'Frontal':
        areas = [28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81]
    elif lobe == 'Central':
        areas = [20, 21, 24, 25, 26, 27, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46]
    elif lobe == 'Appendix':
        areas = [17, 18, 19, 44, 49, 74, 75]

    if lateralisation == 'Left':
        areas = [num for num in areas if num % 2 == 0]
        if lobe == 'Central':
            areas = areas.extend([19, 49])
            areas.sort()
    elif lateralisation == 'Right':
        areas = [num for num in areas if num % 2 != 0]
        if lobe == 'Central':
            areas = areas.extend([44])
            areas.sort()

    return areas


def remove_areas(Y, areas_to_keep):
    areas_to_replace = np. unique(Y)  # list(range(84))
    areas_to_replace = [area for area in areas_to_replace if area not in areas_to_keep]
    for area in areas_to_replace:
        Y = np.where(Y == area, 0, Y)

    return Y


def lobe_selection(lobe):
    if lobe == 'OccipitalParietal':
        areas = [22, 23, 32, 33, 60, 61, 62, 63, 64, 65, 66, 67]
    elif lobe == 'Appendix':
        areas = [17, 18, 19, 44, 49, 74, 75]
    elif 'Frontal' in lobe:
        areas = [28, 29, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81]
    elif 'Central' in lobe:
        areas = [20, 21, 24, 25, 26, 27, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46]
    elif 'Temporal' in lobe:
        areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 30, 31, 47, 48, 82, 83]
    if 'Left' in lobe:
        areas = [num for num in areas if num % 2 == 0]
        # if lobe == 'Central':
        #     areas = areas.extend([19, 49])
        #     areas.sort()
    if 'Right' in lobe:
        areas = [num for num in areas if num % 2 != 0]
        # if lobe == 'Central':
        #     areas = areas.extend([44])
        #     areas.sort()

    return areas


def areas_to_lobes(Y):
    '''
    Turns the labels from 83 regions to 8 labels of the lobes
    '''
    # Temporal right
    areas = [1, 3, 5, 7, 9, 11, 13, 15, 31, 47, 83]
    for area in areas:
        Y = np.where(Y == area, 1, Y)
    # Temporal left
    areas = [2, 4, 6, 8, 10, 12, 14, 16, 30, 48, 82]
    for area in areas:
        Y = np.where(Y == area, 2, Y)
    # OccipitalParietal
    areas = [22, 23, 32, 33, 60, 61, 62, 63, 64, 65, 66, 67]
    for area in areas:
        Y = np.where(Y == area, 3, Y)
    # Frontal left
    areas = [28, 50, 52, 54, 56, 58, 68, 70, 72, 76, 78, 80]
    for area in areas:
        Y = np.where(Y == area, 4, Y)
    # Frontal right
    areas = [29, 51, 53, 55, 57, 59, 69, 71, 73, 77, 79, 81]
    for area in areas:
        Y = np.where(Y == area, 5, Y)
    # Central left
    areas = [20, 24, 26, 34, 36, 38, 40, 42, 46]
    for area in areas:
        Y = np.where(Y == area, 6, Y)
    # Central right
    areas = [21, 25, 27, 35, 37, 39, 41, 43, 45]
    for area in areas:
        Y = np.where(Y == area, 7, Y)
    # Appendix / Remainder
    areas = [17, 18, 19, 44, 49, 74, 75]
    for area in areas:
        Y = np.where(Y == area, 8, Y)

    return Y


def lobes_to_areas(lobe_id):
    if lobe_id == 0:
        label = 'Background'
    if lobe_id == 1:
        label = 'TemporalRight'
    if lobe_id == 2:
        label = 'TemporalLeft'
    if lobe_id == 3:
        label = 'OccipitalParietal'
    if lobe_id == 4:
        label = 'FrontalLeft'
    if lobe_id == 5:
        label = 'FrontalRight'
    if lobe_id == 6:
        label = 'CentralRight'
    if lobe_id == 7:
        label = 'CentralLeft'
    if lobe_id == 8:
        label = 'Appendix'

    if lobe_id == 0:
        areas = 0
    else:
        areas = lobe_selection(label)

    return areas


def retrieve_areas(Y, areas):
    '''Y needs to be in integer coding'''
    empty_Y = np.empty(Y.shape)
    areas_copy = areas.copy()
    areas_copy.insert(0, 0)
    areas_in_segmentation = np.unique(Y)
    a = 0
    for area in areas_copy:
        # print(a, area)
        if a not in areas_in_segmentation:
            print(area, 'not in segmentation')
            a = a + 1
            continue
        else:
            empty_Y = np.where(Y==a, area, empty_Y)
            a = a + 1
    Y_in_areas = empty_Y
    return Y_in_areas


def areas_to_str(area):
    area = str(int(area))
    dict = {
        '0': 'Background',
        '1': 'Hippocampus right',
        '2': 'Hippocampus left',
        '3': 'Amygdala right',
        '4': 'Amygdala left',
        '5': 'Anterior temporal lobe, medial part right',
        '6': 'Anterior temporal lobe, medial part left',
        '7': 'Anterior temporal lobe, lateral part right',
        '8': 'Anterior temporal lobe, lateral part left',
        '9': 'Gyri parahippocampalis et ambiens right',
        '10': 'Gyri parahippocampalis et ambiens left',
        '11': 'Superior temporal gyrus, central part right',
        '12': 'Superior temporal gyrus, central part left',
        '13': 'Medial and inferior temporal gyri right',
        '14': 'Medial and inferior temporal gyri left',
        '15': 'Lateral occipitotemporal gyrus (gyrus fusiformis) right',
        '16': 'Lateral occipitotemporal gyrus (gyrus fusiformis) left',
        '17': 'Cerebellum right',
        '18': 'Cerebellum left',
        '19': 'Brainstem',
        '20': 'Insula left',
        '21': 'Insula right',
        '22': 'Lateral remainder of occipital lobe left',
        '23': 'Lateral remainder of occipital lobe right',
        '24': 'Cingulate gyrus, anterior (supragenual) part left',
        '25': 'Cingulate gyrus, anterior (supragenual) part right',
        '26': 'Cingulate gyrus, posterior part left',
        '27': 'Cingulate gyrus, posterior part right',
        '28': 'Middle frontal gyrus left',
        '29': 'Middle frontal gyrus right',
        '30': 'Posterior temporal lobe left',
        '31': 'Posterior temporal lobe right',
        '32': 'Remainder of parietal lobe left (including supramarginal and angular gyrus)',
        '33': 'Remainder of parietal lobe right (including supramarginal and angular gyrus)',
        '34': 'Caudate nucleus left',
        '35': 'Caudate nucleus right',
        '36': 'Nucleus accumbens left',
        '37': 'Nucleus accumbens right',
        '38': 'Putamen left',
        '39': 'Putamen right',
        '40': 'Thalamus left',
        '41': 'Thalamus right',
        '42': 'Pallidum (globus pallidus) left',
        '43': 'Pallidum (globus pallidus) right',
        '44': 'Corpus callosum',
        '45': 'Lateral ventricle, frontal horn, central part, and occipital horn right',
        '46': 'Lateral ventricle, frontal horn, central part, and occipital horn left',
        '47': 'Lateral ventricle, temporal horn right',
        '48': 'Lateral ventricle, temporal horn left',
        '49': 'Third ventricle',
        '50': 'Precentral gyrus left',
        '51': 'Precentral gyrus right',
        '52': 'Straight gyrus (gyrus rectus) left',
        '53': 'Straight gyrus (gyrus rectus) right',
        '54': 'Anterior orbital gyrus left',
        '55': 'Anterior orbital gyrus right',
        '56': 'Inferior frontal gyrus left',
        '57': 'Inferior frontal gyrus right',
        '58': 'Superior frontal gyrus left',
        '59': 'Superior frontal gyrus right',
        '60': 'Postcentral gyrus left',
        '61': 'Postcentral gyrus right',
        '62': 'Superior parietal gyrus left',
        '63': 'Superior parietal gyrus right',
        '64': 'Lingual gyrus left',
        '65': 'Lingual gyrus right',
        '66': 'Cuneus left',
        '67': 'Cuneus right',
        '68': 'Medial orbital gyrus left',
        '69': 'Medial orbital gyrus right',
        '70': 'Lateral orbital gyrus left',
        '71': 'Lateral orbital gyrus right',
        '72': 'Posterior orbital gyrus left',
        '73': 'Posterior orbital gyrus right',
        '74': 'Substantia nigra left',
        '75': 'Substantia nigra right',
        '76': 'Subgenual anterior cingulate gyrus left',
        '77': 'Subgenual anterior cingulate gyrus right',
        '78': 'Subcallosal area left',
        '79': 'Subcallosal area right',
        '80': 'Pre-subgenual anterior cingulate gyrus left',
        '81': 'Pre-subgenual anterior cingulate gyrus right',
        '82': 'Superior temporal gyrus, anterior part left',
        '83': 'Superior temporal gyrus, anterior part right'
    }

    return dict[area]


def np_dim(array):
    np_shape = array.shape
    dim = len(np_shape)

    return dim


def test_loader(test_set, test_path, test_idx):
    if test_set == 'ADNI':
        X_i = nib.load("".join(test_path + '/' + test_idx + 'cropbrain.nii.gz'))
        Y_i = nib.load("".join(test_path + '/' + test_idx + 'cropseg.nii.gz'))
    if test_set == 'Hammers':
        X_i = nib.load("".join(test_path + '/a' + test_idx + '.nii.gz'))
        Y_i = nib.load("".join(test_path + '/a' + test_idx + '-seg.nii.gz'))
    X_i = np.asarray(X_i.get_data())
    print()
    X_i = np.expand_dims(X_i, axis=4)
    Y_i = np.asarray(Y_i.get_data())

    return X_i, Y_i


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)