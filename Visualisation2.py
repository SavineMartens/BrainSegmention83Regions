import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from Shortlist import *
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib

title_list = ['Axial', 'Sagittal', 'Coronal']
axis_sum1 = [0, 1, 2]
axis_sum2 = [0, -1, 0]


class AnatomicalPlaneViewer(object):
    def __init__(self, X, Y_true, Y_pred):
        'Initialization, make them 3 dimensions'
        self.X = X
        self.Y_true = Y_true
        self.Y_pred = Y_pred

    def select_slice(self, labels, indices, dice):
        if labels == 'gt' or labels == 'true' or labels == 'Y_true':
            Y = self.Y_true
        if labels == 'predicted' or labels == 'pred' or labels == 'Y_pred' or labels == 'prediction':
            Y = self.Y_pred
        axial_index = indices[0]
        sagittal_index = indices[1]
        coronal_index = indices[2]
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:black')
        if dice:
            plt.figtext(0.1, 0.9, 'Dice coefficient: ' + str(dice), color='white')
        for p in [0, 1, 2]:
            if p == 0:
                img_lab = np.squeeze(Y[:, :, axial_index])
                img = self.X[:, :, axial_index]
            elif p == 1:
                img_lab = Y[sagittal_index, :, :]
                img = self.X[sagittal_index, :, :]
            elif p == 2:
                img_lab = np.squeeze(Y[:, coronal_index, :])
                img = self.X[:, coronal_index, :]
            plt.subplot(1, 3, p + 1)
            plt.title(title_list[p], color='white')
            plt.imshow(img, cmap="gray")
            plt.imshow(img_lab, alpha=0.4)

    def max_of_slice(self, labels, dice):
        if labels == 'gt' or labels == 'true' or labels == 'Y_true':
            Y = self.Y_true
        if labels == 'predicted' or labels == 'pred' or labels == 'Y_pred' or labels == 'prediction':
            Y = self.Y_pred
        sum_volume = np.where(Y > 1, 1, Y)  # replace all labels with 1 to find slice with most labels on it
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:black')
        if dice:
            plt.figtext(0.1, 0.9, 'Dice coefficient: ' + str(dice), color='white')
        for p in [0, 1, 2]:
            sum_slice = np.sum(sum_volume, axis=axis_sum1[p])
            sum_slice = np.sum(sum_slice, axis=axis_sum2[p])
            index = np.argmax(sum_slice)
            index = index[0]
            if p == 0:
                img_lab = Y[:, :, index]
                img = self.X[:, :, index]
                axial_index = index
            elif p == 1:
                img_lab = np.squeeze(Y[index, :, :])
                img = self.X[index, :, :]
                sagittal_index = index
            elif p == 2:
                img_lab = np.squeeze(Y[:, index, :])
                img = self.X[:, index, :]
                coronal_index = index
            plt.subplot(1, 3, p + 1)
            plt.title(title_list[p], color='white')
            plt.imshow(img, cmap="gray")
            plt.imshow(img_lab, alpha=0.3)

        return axial_index, sagittal_index, coronal_index

    def legend(self, label_str):
        '''make Y_pred same as Y_true in case you only have ground truth'''
        if label_str == 'gt' or label_str == 'true' or label_str == 'Y_true':
            Y = self.Y_true
        if label_str == 'predicted' or label_str == 'pred' or label_str == 'Y_pred' or label_str == 'prediction':
            Y = self.Y_pred
        sum_volume = np.where(Y > 1, 1, Y)  # replace all labels with 1 to find slice with most labels on it
        values = np.unique(self.Y_true)
        list_legend = [areas_to_str(label) for label in values]
        fig = plt.figure()
        fig.patch.set_facecolor('xkcd:black')
        # if dice:
        #     plt.figtext(0.7, 0.9, 'Dice coefficient: ' + str(dice), color='white')
        for p in [0, 1, 2]:
            sum_slice = np.sum(sum_volume, axis=axis_sum1[p])
            sum_slice = np.sum(sum_slice, axis=axis_sum2[p])
            index = np.argmax(sum_slice)
            if p == 0:
                img_lab = Y[:, :, index]
                img = self.X[:, :, index]
            elif p == 1:
                img_lab = np.squeeze(Y[index, :, :])
                img = self.X[index, :, :]
            elif p == 2:
                img_lab = np.squeeze(Y[:, index, :])
                img = self.X[:, index, :]
            plt.subplot(3, 1, p + 1)
            plt.title(title_list[p], color='white')
            plt.imshow(img, cmap="gray")
            plt.imshow(img_lab, alpha=0.3, label=list_legend)
        # colors = [lab.cmap(lab.norm(value)) for value in values]
        plt.legend(loc=1.1)
        # fig.tight_layout()
        # fig.subplots_adjust(right=0.75)
        # fig, ax = plt.subplots()
        # custom_lines = self.__custom_legend(colors)
        # fig.legend(lab, list_legend)
        plt.show()

    def legend2(self, label_str):
        if label_str == 'gt' or label_str == 'true' or label_str == 'Y_true':
            Y = self.Y_true
        if label_str == 'predicted' or label_str == 'pred' or label_str == 'Y_pred' or label_str == 'prediction':
            Y = self.Y_pred
        sum_volume = np.where(Y > 1, 1, Y)  # replace all labels with 1 to find slice with most labels on it
        values = np.unique(self.Y_true)
        k = len(values)
        print(k, 'locations')
        k_ticks = np.linspace(0.5, k+0.5, k-1)
        c = cmap_discretize('jet', k)
        list_legend = [areas_to_str(label) for label in values]
        print(len(list_legend), 'legend entries')
        fig, axes = plt.subplots(nrows=3, ncols=1)
        # fig.patch.set_facecolor('xkcd:black')
        for p, ax in enumerate(axes.flat):
            sum_slice = np.sum(sum_volume, axis=axis_sum1[p])
            sum_slice = np.sum(sum_slice, axis=axis_sum2[p])
            index = np.argmax(sum_slice)
            if p == 0:
                img_lab = Y[:, :, index]
                img = self.X[:, :, index]
            elif p == 1:
                img_lab = np.squeeze(Y[index, :, :])
                img = self.X[index, :, :]
            elif p == 2:
                img_lab = np.squeeze(Y[:, index, :])
                img = self.X[:, index, :]
            # plt.title(title_list[p], color='white')
            ax.imshow(img, cmap="gray")
            im = ax.imshow(img_lab, cmap=c, alpha=0.3)
        # fig.subplots_adjust(right=0.7)
        cbar_ax = fig.add_axes([0.65, 0.15, 0.05, 0.7])
        cb = fig.colorbar(im, cax=cbar_ax, ticks=k_ticks)
        cb.ax.set_yticklabels(list_legend)
        # cb.update_normal()
        # cb.ax.set_yticks(ticks=k_ticks)
        plt.show()


    def __custom_legend(self, colors):
        custom_lines = []
        for i in np.arange(len(colors)):
            custom_lines.append([Line2D([0], [0], color=(colors[i][0], colors[i][1], colors[i][2]), lw=5)])
        return custom_lines

    def show_differences(self, dice, avd):
        empty_volume = np.zeros(self.X.shape)
        difference = np.where(self.Y_true != self.Y_pred, 1, empty_volume)

        true_copy = np.where(self.Y_true > 1, 1,
                             self.Y_true)  # replace all labels with 1 to find slice with most labels on it
        pred_copy = np.where(self.Y_pred > 1, 1, self.Y_pred)

        fig = plt.figure(figsize=[6.7, 3.3])
        gs1 = gridspec.GridSpec(3, 1)
        gs1.update(wspace=0.005, hspace=0.0)  # set the spacing between axes.
        fig.patch.set_facecolor('xkcd:black')
        for p in [0, 1, 2]:
            # best slice for both
            true_sum_volume = true_copy.copy()
            pred_sum_volume = pred_copy.copy()
            true_sum_slice = np.sum(true_sum_volume, axis=axis_sum1[p])
            pred_sum_slice = np.sum(pred_sum_volume, axis=axis_sum1[p])
            true_sum_slice = np.sum(true_sum_slice, axis=axis_sum2[p])
            pred_sum_slice = np.sum(pred_sum_slice, axis=axis_sum2[p])
            true_index = np.argmax(true_sum_slice)
            pred_index = np.argmax(pred_sum_slice)
            index = int((true_index + pred_index) / 2)
            # print(index)
            # overlap_volume = difference.copy()
            # sum_overlap_2D = np.sum(overlap_volume, axis=axis_sum1[p])
            # sum_overlap_slice = np.sum(sum_overlap_2D, axis=axis_sum2[p])
            # index = np.argmax(sum_overlap_slice)

            if p == 0:
                diff = difference[:, :, index]
                img_lab = self.Y_true[:, :, index]
                img = self.X[:, :, index]
            elif p == 1:
                diff = np.squeeze(difference[index, :, :])
                img_lab = np.squeeze(self.Y_true[index, :, :])
                img = self.X[index, :, :]
            elif p == 2:
                diff = np.squeeze(difference[:, index, :])
                img_lab = np.squeeze(self.Y_true[:, index, :])
                img = self.X[:, index, :]
            # show difference labels
            if dice:
                plt.figtext(0.1, 0.2, 'DSC: ' + str(round(dice, 3)), color='white')
            if avd:
                plt.figtext(0.1, 0.1, 'AVD: ' + str(round(np.mean(avd), 3)) + ' (' + str(round(np.std(avd), 3)) + ')',
                            color='white')
            plt.subplot(1, 3, p + 1)
            plt.title(title_list[p], color='white')
            pwargs = {'interpolation': 'nearest'}
            plt.imshow(img, cmap="gray")
            img_lab[img_lab == 0] = np.nan
            plt.imshow(img_lab, alpha=0.3, cmap='Blues')
            diff[diff == 0] = np.nan
            plt.imshow(diff, alpha=0.35, cmap=plt.cm.hsv, **pwargs)


    def show_overlap(self, id_scan='not provided'):
        '''
        in this case make Y_true prediction of one lobe and Y_pred of another lobe
        '''
        labels1 = self.Y_true
        labels2 = self.Y_pred
        if len(np.unique(labels1) != 2):
            labels1 = np.where(labels1 > 1, 1, labels1)  # replace all labels with 1
        if len(np.unique(labels2) != 2):
            labels2 = np.where(labels2 > 1, 1, labels2)  # replace all labels with 1
        # if not id_scan:
        #     id_scan = 'not provided'

        labels1 = np.where(labels1 == 0, 0.001, labels1)
        labels2 = np.where(labels2 == 0, 0.002, labels2)

        empty_volume = np.zeros(self.X.shape)
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
                img = self.X[:, :, index]
            elif p == 1:
                overlap = np.squeeze(overlap_volume[index, :, :])
                img_lab1 = np.squeeze(labels1[index, :, :])
                img_lab2 = np.squeeze(labels2[index, :, :])
                img = self.X[index, :, :]
            elif p == 2:
                overlap = np.squeeze(overlap_volume[:, index, :])
                img_lab1 = np.squeeze(labels1[:, index, :])
                img_lab2 = np.squeeze(labels2[:, index, :])
                img = self.X[:, index, :]
            # show difference labels
            plt.subplot(1, 3, p + 1)
            plt.title(title_list[p], color='white')
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
        plt.figtext(0.1, 0.1, str(int(overlapping_volume.sum())) + ' voxels are overlapping, ID:' + id_scan,
                    color='white')
        n_overlapping_voxels = int(overlapping_volume.sum())
        location = np.where(overlapping_volume > 0)

        return n_overlapping_voxels, location, overlapping_volume


class view_3D(object):
    def __init__(self, Y_true, Y_pred):
        'Initialization, make them 3 dimensions'
        Y_true.self = Y_true
        Y_pred.self = Y_pred

    def plot1label(self, labels):
        """
        plots labels in 3D
        """
        if labels == 'gt' or labels == 'true' or labels == 'Y_true':
            Y = self.Y_true
        if labels == 'predicted' or labels == 'pred' or labels == 'Y_pred' or labels == 'prediction':
            Y = self.Y_pred
        num_area = np.unique(Y)
        ref_shape = [Y.shape[0], Y.shape[1], Y.shape[2]]
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

        for a in np.arange(1, num_area + 1):
            loc = np.where(Y == a)
            ax.scatter3D(loc[0], loc[1], loc[2], marker=".", alpha=0.9)

        plt.show()

    def plot2labels(self):
        """
        compares predicted labels with true labels in 3D of 2 images
        input: [x, y, z]
        """
        ref_shape = [self.Y_true.shape[1], self.Y_true.shape[2], self.Y_true[3]]
        true_loc = np.where(self.Y_true == 1)
        pred_loc = np.where(self.Y_pred == 1)
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

    def show_difference(self):
        """
           compares predicted labels with true labels in 3D
        """
        ref_shape = [self.Y_true.shape[0], self.Y_true.shape[1], self.Y_true.shape[2]]
        true_lab = self.Y_true
        true_loc = np.where(true_lab == 1)
        pred_lab = self.Y_pred
        pred_loc = np.where(pred_lab == 1)
        volume = self.X
        voxels = ~(volume == 0)

        fig = plt.figure()
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


def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

from matplotlib.transforms import BboxBase as bbase

def squeeze_fig_aspect(fig, preserve='h'):
    preserve = preserve.lower()
    bb = bbase.union([ax.bbox for ax in fig.axes])

    w, h = fig.get_size_inches()
    if preserve == 'h':
        new_size = (h * bb.width / bb.height, h)
    elif preserve == 'w':
        new_size = (w, w * bb.height / bb.width)
    else:
        raise ValueError(
            'preserve must be "h" or "w", not {}'.format(preserve))
    fig.set_size_inches(new_size, forward=True)
