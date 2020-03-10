import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Shortlist import *
import matplotlib.gridspec as gridspec


def max_slice_of_planes_viewer(X_data, Y_data, visualisation):
    """""
    X_data.shape should be [index image, x, y, z] and should consist of all images
    visualisation = boolean
    """""
    axis_sum1 = [0, 1, 2]
    axis_sum2 = [0, -1, 0]
    title_list = ['Axial', 'Sagittal', 'Coronal']
    sample_len = X_data.shape[0]
    sum_gt_total = []
    axial_index = []
    sagittal_index = []
    coronal_index = []
    for i in np.arange(0, sample_len):
        volume = X_data[i, ]
        label3D = Y_data[i, ]
        # if visualisation:
        #     fig = plt.figure(i)
        #     fig.patch.set_facecolor('xkcd:black')
        #     plt.figtext(0.1, 0.1, 'Subject number: ' + str(i+1), color='white')
        for p in [0, 1, 2]:
            sum_slice = np.sum(label3D, axis=axis_sum1[p])
            sum_slice = np.sum(sum_slice, axis=axis_sum2[p])
            index = np.argmax(sum_slice)
            if p == 0:
                img_lab = np.squeeze(label3D[:, :, index])
                img = volume[:, :, index]
                axial_index.append(index)
            elif p == 1:
                img_lab = np.squeeze(label3D[index, :, :])
                img = volume[index, :, :]
                sagittal_index.append(index)
            elif p == 2:
                img_lab = np.squeeze(label3D[:, index, :])
                img = volume[:, index, :]
                coronal_index.append(index)
            if visualisation:
                fig = plt.figure(i)
                fig.patch.set_facecolor('xkcd:black')
                plt.figtext(0.1, 0.1, 'Subject number: ' + str(i + 1), color='white')
                plt.subplot(1, 3, p + 1)
                plt.title(title_list[p], color='white')
                plt.imshow(img, cmap="gray")
                plt.imshow(img_lab, alpha=0.3)
        sum_gt_total.append(np.sum(sum_slice))
        if visualisation:
            plt.show()

    return axial_index, sagittal_index, coronal_index, sum_gt_total


def max_slice_of_planes_viewer_mc(X_data, Y_data, visualisation):
    """""
    X_data.shape should be [index imtitle_list = ['Axial', 'Sagittal', 'Coronal']age, x, y, z] and should consist of all images
    For multiclass data
    visualisation = boolean
    """""
    axis_sum1 = [0, 1, 2]
    axis_sum2 = [0, -1, 0]
    title_list = ['Axial', 'Sagittal', 'Coronal']
    sample_len = X_data.shape[0]
    sum_gt_total = []
    axial_index = []
    sagittal_index = []
    coronal_index = []
    copy = np.where(Y_data > 1, 1, Y_data)  # replace all labels with 1 to find slice with most labels on it
    for i in np.arange(0, sample_len):
        volume = X_data[i, ]
        label3D = Y_data[i, ]
        sum_volume = copy[i, ]
        if visualisation:
            fig = plt.figure()
        for p in [0, 1, 2]:
            sum_slice = np.sum(sum_volume, axis=axis_sum1[p])
            sum_slice = np.sum(sum_slice, axis=axis_sum2[p])
            index = np.argmax(sum_slice)
            if p == 0:
                img_lab = label3D[:, :, index]
                img = volume[:, :, index]
                axial_index.append(index)
            elif p == 1:
                img_lab = np.squeeze(label3D[index, :, :])
                img = volume[index, :, :]
                sagittal_index.append(index)
            elif p == 2:
                img_lab = np.squeeze(label3D[:, index, :])
                img = volume[:, index, :]
                coronal_index.append(index)
            if visualisation:
                fig.patch.set_facecolor('xkcd:black')
                plt.figtext(0.1, 0.1, 'Subject number: ' + str(i + 1), color='white')
                plt.subplot(1, 3, p + 1)
                plt.title(title_list[p], color='white')
                plt.imshow(img, cmap="gray")
                plt.imshow(img_lab, alpha=0.3)
        sum_gt_total.append(np.sum(sum_slice))
        if visualisation:
            plt.show()

    return axial_index, sagittal_index, coronal_index, sum_gt_total


def show_slices(slices, colour_map):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap=colour_map, origin="lower")
    plt.show()

# def show_slices_labels(slices):
#     """ Function to display row of image slices """
#     fig, axes = plt.subplots(1, len(slices))
#     for i, slice in enumerate(slices):
#         axes[i].imshow(slice.T, cmap="gray", origin="lower")
#     plt.show()


def predict_slice_viewer(X_data, Y_pred_data, axial_array, sagittal_array, coronal_array, results):
    """""
    X_data array of size [index, x, y, z ]
    Y_pred_data of size [index, x, y, z]
    """""
    title_list = ['Axial', 'Sagittal', 'Coronal']
    sample_len = X_data.shape[0]
    for i in np.arange(0, sample_len):
        volume = np.squeeze(X_data[i, ])
        label3D = Y_pred_data[i, ]
        fig = plt.figure(i)
        fig.patch.set_facecolor('xkcd:black')
        plt.figtext(0.1, 0.1, 'Predicted segmentation for test sample No.: ' + str(i + 1), color='white')
        if results:
            plt.figtext(0.1, 0.9, 'Dice coefficient: ' + str(results[i]), color='white')
        axial_index = axial_array[i]
        sagittal_index = sagittal_array[i]
        coronal_index = coronal_array[i]
        for p in [0, 1, 2]:
            if p == 0:
                img_lab = np.squeeze(label3D[:, :, axial_index])
                img = volume[:, :, axial_index]
            elif p == 1:
                img_lab = label3D[sagittal_index, :, :]
                img = volume[sagittal_index, :, :]
            elif p == 2:
                img_lab = np.squeeze(label3D[:, coronal_index, :])
                img = volume[:, coronal_index, :]
            plt.subplot(1, 3, p + 1)
            plt.title(title_list[p], color='white')
            plt.imshow(img, cmap="gray")
            plt.imshow(img_lab, alpha=0.4)
        plt.show()


def plot_3D(Y_data, num_area):
    """
    plots labels in 3D
    """
    ref_shape = [Y_data.shape[0], Y_data.shape[1], Y_data.shape[2]]
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    axl = plt.gca()
    axl.set_xlim3d([0, ref_shape[0]])
    axl.set_ylim3d([0, ref_shape[1]])
    axl.set_zlim3d([0, ref_shape[2]])

    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.set_xlabel('Width', c='white')
    ax.set_ylabel('Depth', c='white')
    ax.set_zlabel('Height', c='white')

    for a in np.arange(1, num_area+1):
        loc = np.where(Y_data == a)
        ax.scatter3D(loc[0], loc[1], loc[2], marker=".", alpha=0.9)

    plt.show()


def plot_3D_compare(true_lab, pred_lab):
    """
    compares predicted labels with true labels in 3D of 2 images
    input: [x, y, z]
    """
    ref_shape = [true_lab.shape[1], true_lab.shape[2], true_lab.shape[3]]
    true_loc = np.where(true_lab == 1)
    pred_loc = np.where(pred_lab == 1)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    axl = plt.gca()
    axl.set_xlim3d([0, ref_shape[0]])
    axl.set_ylim3d([0, ref_shape[1]])
    axl.set_zlim3d([0, ref_shape[2]])
    ax.scatter3D(true_loc[0], true_loc[1], true_loc[2], marker=".", alpha=0.9)
    ax.scatter3D(pred_loc[0], pred_loc[1], pred_loc[2], marker=".", alpha=0.05)

    fig.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.set_xlabel('Width', c='white')
    ax.set_ylabel('Depth', c='white')
    ax.set_zlabel('Height', c='white')

    plt.show()


def plot_3D_compare_list(Y_data_test_list, Y_pred_data_list, ref_shape):
    """
    compares predicted labels with true labels in 3D of list of images
    has to be shape: [index, x, y, z]
    """
    sample_len = Y_data_test_list.shape[0]
    num_classes = Y_data_test_list.max()+1
    for i in np.arange(0, sample_len):
        for c in np.arange(1, num_classes):
            fig = plt.figure()
            plt.figtext(0.1, 0.1, 'Predicted segmentation for test sample No.: ' + str(i + 1) + ', class: ' + str(c), color='white')
            true_lab = Y_data_test_list[i, ]
            true_loc = np.where(true_lab == c)
            pred_lab = Y_pred_data_list[i, ]
            pred_loc = np.where(pred_lab == c)

            # concurring locations
            true_copy = true_lab.copy()
            np.place(true_copy, true_copy != c, 99)
            pred_copy = pred_lab.copy()
            np.place(pred_copy, pred_copy != c, 98)
            same_loc = np.where(true_copy == pred_copy)

            ax = plt.axes(projection="3d")
            axl = plt.gca()
            axl.set_xlim3d([0, ref_shape[0]])
            axl.set_ylim3d([0, ref_shape[1]])
            axl.set_zlim3d([0, ref_shape[2]])
            ax.scatter3D(true_loc[0], true_loc[1], true_loc[2], marker=".", alpha=0.2,
                         edgecolor="dodgerblue", facecolor="dodgerblue")
            ax.scatter3D(pred_loc[0], pred_loc[1], pred_loc[2], marker=".", alpha=0.01,
                         edgecolor="lightcoral", facecolor="lightcoral")
            ax.scatter3D(same_loc[0], same_loc[1], same_loc[2], marker=".", alpha=1,
                         edgecolor="white", facecolor="white")

            fig.set_facecolor('black')
            ax.set_facecolor('black')
            ax.grid(False)
            ax.w_xaxis.pane.fill = False
            ax.w_yaxis.pane.fill = False
            ax.w_zaxis.pane.fill = False

            ax.set_xlabel('Width', c='white')
            ax.set_ylabel('Depth', c='white')
            ax.set_zlabel('Height', c='white')

            plt.show()



def plot_3D_accuracy_list(Y_data_test_list, Y_pred_data_list, ref_shape):
    """
    compares predicted labels with true labels in 3D of list of images
    has to be shape: [index, x, y, z]
    """
    sample_len = Y_data_test_list.shape[0]
    num_classes = int(Y_data_test_list.max()+1)
    for i in np.arange(0, sample_len):
        for c in np.arange(1, num_classes):
            fig = plt.figure()
            plt.figtext(0.1, 0.1, 'Predicted segmentation for test sample No.: ' + str(i + 1) + ', class: ' + str(c), color='white')
            true_lab = Y_data_test_list[i, ]
            # true_loc = np.where(true_lab == c)
            pred_lab = Y_pred_data_list[i, ]
            # pred_loc = np.where(pred_lab == c)

            # concurring locations
            true_copy = true_lab.copy()
            np.place(true_copy, true_copy != c, 99)
            pred_copy = pred_lab.copy()
            same_loc = np.where(true_copy == pred_copy)
            np.place(pred_copy, pred_copy != c, 99)
            not_same_loc = np.where(true_copy != pred_copy)

            ax = plt.axes(projection="3d")
            axl = plt.gca()
            axl.set_xlim3d([0, ref_shape[0]])
            axl.set_ylim3d([0, ref_shape[1]])
            axl.set_zlim3d([0, ref_shape[2]])
            ax.scatter3D(not_same_loc[0], not_same_loc[1], not_same_loc[2], marker=".", alpha=0.01,
                         edgecolor="lightcoral", facecolor="lightcoral")
            ax.scatter3D(same_loc[0], same_loc[1], same_loc[2], marker=".", alpha=1,
                         edgecolor="white", facecolor="white")

            fig.set_facecolor('black')
            ax.set_facecolor('black')
            ax.grid(False)
            ax.w_xaxis.pane.fill = False
            ax.w_yaxis.pane.fill = False
            ax.w_zaxis.pane.fill = False

            ax.set_xlabel('Width', c='white')
            ax.set_ylabel('Depth', c='white')
            ax.set_zlabel('Height', c='white')

            plt.show()




def plot_3D_compare_voxels(Y_data_test, Y_pred_data, X_data_test, ref_shape):
    """
    compares predicted labels with true labels in 3D
    """
    sample_len = Y_data_test.shape[0]
    for i in np.arange(0, sample_len):
        true_lab = Y_data_test[i, ]
        true_loc = np.where(true_lab == 1)
        pred_lab = Y_pred_data[i, ]
        pred_loc = np.where(pred_lab == 1)
        volume = X_data_test[i, ]
        voxels = ~(volume==0)
        fig = plt.figure(i)
        ax = plt.axes(projection="3d")
        axl = plt.gca()
        axl.set_xlim3d([0, ref_shape[0]])
        axl.set_ylim3d([0, ref_shape[1]])
        axl.set_zlim3d([0, ref_shape[2]])
        vx = fig.gca(projection='3d')
        vx.voxels(voxels, facecolors=volume, edgecolor='k')
        ax.scatter3D(true_loc[0], true_loc[1], true_loc[2], marker=".", alpha=0.9)
        ax.scatter3D(pred_loc[0], pred_loc[1], pred_loc[2], marker=".", alpha=0.05)

        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.grid(False)
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False

        ax.set_xlabel('Width', c='white')
        ax.set_ylabel('Depth', c='white')
        ax.set_zlabel('Height', c='white')

        plt.show()

# # make dummy data for following function
# import numpy as np
# import nibabel as nib
# from Shortlist import *
# import matplotlib.pyplot as plt
# # from Visualisation import show_differences
# test_path = '/home/jara/Savine/datasize176_208_160/Hammers'
# ID =str(27)
# X_i = nib.load("".join(test_path + '/a' + ID + '.nii.gz'))
# X_i = np.asarray(X_i.get_data())
# scan = np.expand_dims(X_i, axis=4)
# Y_i = nib.load("".join(test_path + '/a' + ID + '-seg.nii.gz'))
# Y_i = np.asarray(Y_i.get_data())
# areas = area_selection('Frontal', 'Left')
# true_labels = remove_areas(Y_i, areas)
# # true_labels = encode1hot(true_labels)
# pred_labels = np.load('/home/jara/Savine/datasize176_208_160/hyperparametertuning/ADNI_200/Frontal/LeftInitial_LRE200/PredictedSegmentationADNI_2000.001/Y_pred_data2020-01-17_06:47:32.npy')
# scan = X_i
# dice = 0.0
#
# max_slice_of_planes_viewer_mc(np.expand_dims(X_i, axis=0), np.expand_dims(true_labels, axis=0), True)
# max_slice_of_planes_viewer_mc(np.expand_dims(X_i, axis=0), pred_labels, True)
#
# show_differences(scan, true_labels, pred_labels, dice, 'white')


def show_differences(scan, true_labels, pred_labels, dice, text_colour):
    '''

    :param scan: mri scan of subject, grayscale, size: [1, x, y, z]
    :param true_labels: volume with (ground truth) labels of scan, integer labels >1, size: [1, x, y, z]
    :param pred_labels: volume with labels of scan, encoded 1 hot, size: [x, y, z, labels]
    :param dice: dice score
    :param text_colour: string colour, black if on cluster, white if on computer
    :return: figure with the different voxels
    '''

    axis_sum1 = [0, 1, 2]
    axis_sum2 = [0, -1, 0]
    title_list = ['Axial', 'Sagittal', 'Coronal']

    # make all same size
    if true_labels.ndim != 3:
        if len(np.unique(true_labels)) == 2:
            true_labels = np.argmax(true_labels, axis=-1)
        else:
            true_labels = np.squeeze(true_labels)
    if scan.ndim != 3:
        scan = np.squeeze(scan)
    if pred_labels.ndim != 3:
        if len(np.unique(pred_labels)) == 2:
            pred_labels = np.argmax(pred_labels, axis=-1)
        else:
            pred_labels = np.squeeze(pred_labels)
    if pred_labels.max() != true_labels.max():
        pred_labels = labels_from_onehot(pred_labels, np.unique(true_labels)[1:-1])
    # difference
    vol_shape = scan.shape
    empty_volume = np.zeros(vol_shape)
    difference = np.where(true_labels != pred_labels, 1, empty_volume)

    new_y_true = np.where(difference != 1, 0, true_labels)
    (labels, counts) = np.unique(new_y_true, return_counts=True)
    sorted_counts, sorted_labels = (list(t) for t in zip(*sorted(zip(counts, labels))))

    true_copy = np.where(true_labels > 1, 1, true_labels)  # replace all labels with 1 to find slice with most labels on it
    pred_copy = np.where(pred_labels > 1, 1, pred_labels)
    true_sum_volume = true_copy
    pred_sum_volume = pred_copy

    fig = plt.figure()
    fig.patch.set_facecolor('xkcd:black')
    for p in [0, 1, 2]:
        # best slice for both
        true_sum_slice = np.sum(true_sum_volume, axis=axis_sum1[p])
        pred_sum_slice = np.sum(pred_sum_volume, axis=axis_sum1[p])
        true_sum_slice = np.sum(true_sum_slice, axis=axis_sum2[p])
        pred_sum_slice = np.sum(pred_sum_slice, axis=axis_sum2[p])
        true_index = np.argmax(true_sum_slice)
        pred_index = np.argmax(pred_sum_slice)
        index = int((true_index+pred_index)/2)
        # print(index)
        # overlap_volume = difference.copy()
        # sum_overlap_2D = np.sum(overlap_volume, axis=axis_sum1[p])
        # sum_overlap_slice = np.sum(sum_overlap_2D, axis=axis_sum2[p])
        # index = np.argmax(sum_overlap_slice)

        if p == 0:
            diff = difference[:, :, index]
            img_lab = true_labels[:, :, index]
            img = scan[:, :, index]
        elif p == 1:
            diff = np.squeeze(difference[index, :, :])
            img_lab = np.squeeze(true_labels[index, :, :])
            img = scan[index, :, :]
        elif p == 2:
            diff = np.squeeze(difference[:, index, :])
            img_lab = np.squeeze(true_labels[:, index, :])
            img = scan[:, index, :]
        # show difference labels
        plt.figtext(0.1, 0.1, 'Dice coefficient:' + str(dice), color=text_colour)
        plt.subplot(1, 3, p + 1)
        plt.title(title_list[p], color=text_colour)
        pwargs = {'interpolation': 'nearest'}
        plt.imshow(img, cmap="gray")
        plt.imshow(img_lab, alpha=0.1, cmap='hot')
        diff[diff == 0] = np.nan
        plt.imshow(diff, alpha=0.35, cmap=plt.cm.hsv, **pwargs)
        fig.patch.set_facecolor('xkcd:black')
    # plt.show()


def show_overlap(scan, labels1, labels2, id, text_colour):
    '''

    :param scan: mri scan of subject, grayscale, size: [1, x, y, z]
    :param labels1: volume with zero for BG, one for FG of one lobe segmentation
    :param labels2: volume with zero for BG, one for FG of another lobe segmentation
    :param text_colour: string colour, black if on cluster, white if on computer
    :return: figure with the different voxels
    '''

    axis_sum1 = [0, 1, 2]
    axis_sum2 = [0, -1, 0]
    title_list = ['Axial', 'Sagittal', 'Coronal']

    # make all same size
    if np_dim(labels1) != 3:
            labels1 = np.squeeze(labels1)
    if np_dim(scan) != 3:
        scan = np.squeeze(scan)
    if np_dim(labels2) != 3:
        labels2 = np.squeeze(labels2)

    if len(np.unique(labels1) != 2):
        labels1 = np.where(labels1 > 1, 1, labels1)  # replace all labels with 1
    if len(np.unique(labels2) != 2):
        labels2 = np.where(labels2 > 1, 1, labels2)  # replace all labels with 1

    labels1 = np.where(labels1 == 0, 0.001, labels1)
    labels2 = np.where(labels2 == 0, 0.002, labels2)

    empty_volume = np.zeros(scan.shape)
    overlapping_volume = np.where(labels1 == labels2, 1, empty_volume)

    fig = plt.figure(figsize=[6.7, 3.3])
    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(wspace=0.005, hspace=0.0)  # set the spacing between axes.
    fig.patch.set_facecolor('xkcd:black')
    for p in [0, 1, 2]:
        # slice with most overlap
        overlap_volume = overlapping_volume.copy()
        sum_overlap_2D = np.sum(overlap_volume, axis=axis_sum1[p])
        sum_overlap_slice = np.sum(sum_overlap_2D, axis=axis_sum2[p])
        index = np.argmax(sum_overlap_slice)
        # print(index)
        if p == 0:
            overlap = overlap_volume[:, :, index]
            img_lab1 = labels1[:, :, index]
            img_lab2 = labels2[:, :, index]
            img = scan[:, :, index]
        elif p == 1:
            overlap = np.squeeze(overlap_volume[index, :, :])
            img_lab1 = np.squeeze(labels1[index, :, :])
            img_lab2 = np.squeeze(labels2[index, :, :])
            img = scan[index, :, :]
        elif p == 2:
            overlap = np.squeeze(overlap_volume[:, index, :])
            img_lab1 = np.squeeze(labels1[:, index, :])
            img_lab2 = np.squeeze(labels2[:, index, :])
            img = scan[:, index, :]
        # show difference labels
        plt.subplot(1, 3, p + 1)
        plt.title(title_list[p], color=text_colour)
        img = np.rot90(img)
        img_lab1 = np.rot90(img_lab1)
        img_lab2 = np.rot90(img_lab2)
        overlap = np.rot90(overlap)
        overlap[overlap == 0] = None
        plt.imshow(img, cmap="gray")
        plt.imshow(img_lab1, alpha=0.3, cmap='hot')
        plt.imshow(img_lab2, alpha=0.3, cmap='copper')
        pwargs = {'interpolation': 'nearest'}
        plt.imshow(overlap, alpha=0.8, cmap=plt.cm.hsv, **pwargs)
        fig.patch.set_facecolor('xkcd:black')
    plt.figtext(0.1, 0.1, str(int(overlapping_volume.sum())) + ' voxels are overlapping, ID:' + id, color=text_colour)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    n_overlapping_voxels = int(overlapping_volume.sum())
    location = np.where(overlapping_volume>0)

    return n_overlapping_voxels, location, overlapping_volume
