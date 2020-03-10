import numpy as np


class Metrics(object):
    ''' Use integer encoding, not one-hot! Will be too big in case of 83 regions'''
    def __init__(self, Y_true, Y_pred):
        self.Y_true = Y_true
        self.Y_pred = Y_pred

    def __check_datatype(self):
        if len(np.unique(self.Y_true)) == 2 or len(np.unique(self.Y_pred)) == 2:
            print('Must use integer encoding, not one-hot-encoding!')

        areas_true = np.unique(self.Y_true)
        areas_pred = np.unique(self.Y_pred)
        if np.sum(areas_true != areas_pred) != 0:
            print('Not comparing the same classes!')
        return areas_true

    def AVD(self):
        '''
        :return: Absolute Volume Difference
        '''
        areas_true = self.__check_datatype()
        AVD_all = []

        for a, area in enumerate(areas_true):
            empty_true = np.zeros(self.Y_true.shape)
            empty_pred = np.zeros(self.Y_pred.shape)
            V_true = np.sum(np.where(self.Y_true == int(area), 1, empty_true))
            V_pred = np.sum(np.where(self.Y_pred == int(area), 1, empty_pred))
            AVD_c = (abs(V_pred - V_true)/V_true)*100
            AVD_all.append(AVD_c)
        mean_AVD = sum(AVD_all) / len(AVD_all)

        return mean_AVD, AVD_all

    def DSC_binary(self):
        '''Binary/Integer input!'''
        areas_true = self.__check_datatype()

        DC_all = []

        for a, area in enumerate(areas_true):
            empty_true = np.zeros(self.Y_true.shape)
            empty_pred = np.zeros(self.Y_pred.shape)
            sel_class_true = np.where(self.Y_true == int(area), 1, empty_true)
            sel_class_pred = np.where(self.Y_pred == int(area), 1, empty_pred)
            numerator = np.sum(sel_class_true * sel_class_pred)
            denominator = np.sum(sel_class_true + sel_class_pred)
            DC = (2 * numerator) / denominator
            DC_all.append(DC)
        mean_DSC = sum(DC_all) / len(DC_all)

        return mean_DSC, DC_all

    def DSC_probability(self):
        '''Input should be probabilities in size of one-hot-encoding!'''
        # How do I get probabilities of true?
        FG_true = self.Y_true[:, :, :, 1:]
        FG_pred = self.Y_pred[:, :, :, 1:]
        intersection = np.sum(FG_true * FG_pred)
        union = np.sum(FG_true) + np.sum(FG_pred)
        dice = np.mean(2. * intersection / union)

        return dice

    def DSC_FG_binary(self):
        areas_true = self.__check_datatype()
        DC_all = []

        for a, area in enumerate(areas_true):
            empty_true = np.zeros(self.Y_true.shape)
            empty_pred = np.zeros(self.Y_pred.shape)
            sel_class_true = np.where(self.Y_true == int(area), 1, empty_true)
            sel_class_pred = np.where(self.Y_pred == int(area), 1, empty_pred)
            numerator = np.sum(sel_class_true * sel_class_pred)
            denominator = np.sum(sel_class_true + sel_class_pred)
            DC = (2 * numerator) / denominator
            DC_all.append(DC)
        mean_DSC = sum(DC_all[1:-1]) / len(DC_all[1:-1])

        return mean_DSC, DC_all

    # # used https://github.com/amanbasu/3d-prostate-segmentation/blob/master/metric_eval.py for the following functions
    # def HD95(self, voxel_spacing=None, connectivity=1):
    #     """
    #     COPIED COMPLETELY FROM SOURCE!
    #    The distances between the surface voxel of binary objects in result and their
    #    nearest partner surface voxel of a binary object in reference.
    #    """
    #     result = np.atleast_1d(result.astype(numpy.bool))
    #     reference = numpy.atleast_1d(reference.astype(numpy.bool))
    #     if voxelspacing is not None:
    #         voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
    #         voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
    #         if not voxelspacing.flags.contiguous:
    #             voxelspacing = voxelspacing.copy()
    #
    #     # binary structure
    #     footprint = generate_binary_structure(result.ndim, connectivity)
    #
    #     # test for emptiness
    #     if 0 == numpy.count_nonzero(result):
    #         raise RuntimeError('The first supplied array does not contain any binary object.')
    #     if 0 == numpy.count_nonzero(reference):
    #         raise RuntimeError('The second supplied array does not contain any binary object.')
    #
    #         # extract only 1-pixel border line of objects
    #     result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    #     reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    #
    #     # compute average surface distance
    #     # Note: scipys distance transform is calculated only inside the borders of the
    #     #       foreground objects, therefore the input has to be reversed
    #     dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    #     sds = dt[result_border]
    #
    #     return sds
    # def HDD

    def precision(self):
        areas_true = self.__check_datatype()
        precision_all = []
        for a, area in enumerate(areas_true):
            empty_true = np.zeros(self.Y_true.shape)
            empty_pred = np.zeros(self.Y_pred.shape)
            sel_class_true = np.sum(np.where(self.Y_true == int(area), 1, empty_true))
            sel_class_pred = np.sum(np.where(self.Y_pred == int(area), 1, empty_pred))
            true_pos = np.count_nonzero(sel_class_pred & sel_class_true)
            false_pos = np.count_nonzero(sel_class_pred & ~sel_class_true)
            try:
                precision_a = true_pos / (true_pos + false_pos)
            except ZeroDivisionError:
                precision_a = 0.0
            precision_all.append(precision_a)
        mean_precision = sum(precision_all/len(precision_all))

        return mean_precision, precision_all

    def recall(self):
        areas_true = self.__check_datatype()
        recall_all = []
        for a, area in enumerate(areas_true):
            empty_true = np.zeros(self.Y_true.shape)
            empty_pred = np.zeros(self.Y_pred.shape)
            sel_class_true = np.sum(np.where(self.Y_true == int(area), 1, empty_true))
            sel_class_pred = np.sum(np.where(self.Y_pred == int(area), 1, empty_pred))
            true_pos = np.count_nonzero(sel_class_pred & sel_class_true)
            false_neg = np.count_nonzero(~sel_class_pred & sel_class_true)
            try:
                precision_a = true_pos / (true_pos + false_neg)
            except ZeroDivisionError:
                precision_a = 0.0
            recall_all.append(precision_a)
        mean_recall = sum(recall_all / len(recall_all))

        return mean_recall, recall_all

