import numpy as np
import keras
import os
import nibabel as nib
from Shortlist import *
import random
from Visualisation import max_slice_of_planes_viewer_mc

"""""
Augmented data is loaded in .nii.gz format. Training size per epoch can be predefined in advance. 
"""""

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, batch_size, dim_x, dim_y, dim_z, n_classes, shuffle=True, verbose=False):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim = (dim_x, dim_y, dim_z)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.verbose = verbose

    def generate_validation(self, part_index, areas, path):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(part_index)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            if len(indexes) % self.batch_size != 0:
                print(len(indexes) % self.batch_size, 'item(s) are not used for generation!')
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [part_index[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                if self.verbose and len(indexes) < 10:
                    print(i+1, '/', imax, 'list of temporary validation IDs: ', list_IDs_temp)
                if self.verbose and len(indexes) > 10:
                    print(i + 1, '/', imax, 'list of temporary training IDs: ', list_IDs_temp)
                # Generate data
                x, y = self.__validation_generation(list_IDs_temp, areas, path)

                yield x, y


    def __get_exploration_order(self, part_index):
        'Generates order of exploration'
        # Find exploration order
        if issubclass(type(part_index), np.ndarray):  # validation + training Baseline + DA_trans
            # Find exploration order
            indices = np.arange(len(part_index))
            if self.shuffle:
                np.random.shuffle(indices)

        if issubclass(type(part_index), list):  # (when) am I using this?
            print('part_index is a list')
            indices = part_index.copy()
            random.shuffle(indices)

        return indices

    def __validation_generation(self, list_IDs_temp, areas, path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1)).astype(dtype='float16') # , dtype=float
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int).astype(dtype='uint8') #  #1

        # Generate validation data
        for i in np.arange(len(list_IDs_temp)):
            # print(i)
            # print('i in loop for data generation: ', i+1, ' out of ', str(self.batch_size))
            ID = str(list_IDs_temp[i]+1)
            ID = ID.zfill(2)
            X_i = nib.load("".join(path + '/a' + ID + '.nii.gz'))
            X_i = np.asarray(X_i.get_data())
            X_i = np.expand_dims(X_i, axis=4)
            Y_i = nib.load("".join(path + '/a' + ID + '-seg.nii.gz'))
            Y_i = np.asarray(Y_i.get_data())
            Y_i = remove_areas(Y_i, areas)
            X[i, ] = X_i
            Y_i = encode1hot(Y_i)
            Y[i, ] = Y_i

        return X, Y


    def generate_training(self, list_training, path, areas, training_size):
        'Generates batches of samples'
        # Infinite loop
        while 1:

            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_training)

            # Generate batches
            imax = int(training_size / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_training[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                if self.verbose:
                    print(i+1, '/', training_size, 'list of temporary training IDs: ', list_IDs_temp)
                # Generate data
                x, y = self.__training_generation(list_IDs_temp, areas, path)

                yield x, y

    def __training_generation(self, list_IDs_temp, areas, path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1)).astype(dtype='float16')
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        areas_to_replace = list(range(84))
        areas_to_replace = [area for area in areas_to_replace if area not in areas]
        # print(list_IDs_temp)
        # Generate validation data
        for i in np.arange(len(list_IDs_temp)):
            # print(i)
            # print('i in loop for data generation: ', i+1, ' out of ', str(self.batch_size))
            ID = list_IDs_temp[i]
            # print(type(ID), ID)
            X_i = nib.load("".join(path + '/a' + ID + '.nii.gz'))
            X_i = np.asarray(X_i.get_data())
            Y_i = nib.load("".join(path + '/a' + ID + '-seg.nii.gz'))
            Y_i = np.asarray(Y_i.get_data())
            X_i = np.expand_dims(X_i, axis=4)
            # Store data in format: [batch, x, y, z, 1/classes]
            X[i, ] = X_i
            for area in areas_to_replace:
                Y_i = np.where(Y_i == area, 0, Y_i)
            Y_temp = encode1hot(Y_i)  # np.expand_dims(Y_i, axis=4)
            Y[i, ] = Y_temp
        return X, Y

    def adni_training(self, list_training, path, areas, training_size):
        'Generates batches of samples'
        # Infinite loop
        while 1:

            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_training)

            # Generate batches
            imax = int(training_size / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_training[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                if self.verbose:
                    print(i + 1, '/', training_size, 'list of temporary training IDs: ', list_IDs_temp)
                # Generate data
                x, y = self.__adni_generation(list_IDs_temp, areas, path)

                yield x, y

    def __adni_generation(self, list_IDs_temp, areas, path):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1)).astype(dtype='float16')
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        areas_to_replace = list(range(84))
        areas_to_replace = [area for area in areas_to_replace if area not in areas]

        for i in np.arange(len(list_IDs_temp)):
            # print(i)
            # print('i in loop for data generation: ', i+1, ' out of ', str(self.batch_size))
            ID = list_IDs_temp[i]
            # print(type(ID), ID)
            X_i = nib.load("".join(path + '/' + ID + 'cropbrain.nii.gz'))
            X_i = np.asarray(X_i.get_data())
            Y_i = nib.load("".join(path + '/' + ID + 'cropseg.nii.gz'))
            Y_i = np.asarray(Y_i.get_data())
            X_i = np.expand_dims(X_i, axis=4)
            # Store data in format: [batch, x, y, z, 1/classes]
            X[i,] = X_i
            for area in areas_to_replace:
                Y_i = np.where(Y_i == area, 0, Y_i)
            # print('areas in Y_i', np.unique(Y_i))
            Y_temp = encode1hot(Y_i) # np.expand_dims(Y_i, axis=4)
            Y[i,] = Y_temp
        return X, Y

    def mixed_training(self, list_training, DA_path, BL_path, areas, training_size):
        'Generates batches of samples'
        # Infinite loop
        # part_index = np.arange(training_size)
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_training)

            # Generate batches
            imax = int(training_size / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_training[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                if self.verbose:
                    print(i + 1, '/', training_size, 'list of temporary training IDs: ', list_IDs_temp)
                # Generate data
                x, y = self.__mixed_generation(list_IDs_temp, areas, DA_path, BL_path)

                yield x, y

    def __mixed_generation(self, list_IDs_temp, areas, DA_path, BL_path):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1)).astype(dtype='float16')
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i in np.arange(len(list_IDs_temp)):
            ID = list_IDs_temp[i]
            if not 'bl' in ID and not '_' in ID:  # Hammers original data
                ID = str(int(ID) + 1).zfill(2)
                path = BL_path
                X_i = nib.load("".join(path + '/a' + ID + '.nii.gz'))
                Y_i = nib.load("".join(path + '/a' + ID + '-seg.nii.gz'))
            if '_' in ID and not 'bl' in ID:  # Hammers Augmented
                path = DA_path
                X_i = nib.load("".join(path + '/a' + ID + '.nii.gz'))
                Y_i = nib.load("".join(path + '/a' + ID + '-seg.nii.gz'))
            if 'bl' in ID:  # ADNI data
                path = DA_path
                X_i = nib.load("".join(path + '/' + ID + 'cropbrain.nii.gz'))
                Y_i = nib.load("".join(path + '/' + ID + 'cropseg.nii.gz'))
            X_i = np.asarray(X_i.get_data())
            Y_i = np.asarray(Y_i.get_data())
            X_i = np.expand_dims(X_i, axis=4)
            # Store data in format: [batch, x, y, z, 1/classes]
            X[i, ] = X_i
            Y_i = remove_areas(Y_i, areas)
            # print('areas in Y_i', np.unique(Y_i), '(', len(np.unique(Y_i)),')')
            Y_temp = encode1hot(Y_i)  # np.expand_dims(Y_i, axis=4)
            Y[i, ] = Y_temp
        return X, Y


    def lobe_training(self, list_training, DA_path, BL_path):
        'Generates batches of samples'
        # Infinite loop
        # part_index = np.arange(training_size)
        training_size = len(list_training)
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_training)

            # Generate batches
            imax = int(training_size / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_training[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                if self.verbose:
                    print(i + 1, '/', training_size, 'list of temporary training IDs: ', list_IDs_temp)
                # Generate data
                x, y = self.__lobe_generation(list_IDs_temp, DA_path, BL_path)

                yield x, y

    def __lobe_generation(self, list_IDs_temp, DA_path, BL_path):
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1)).astype(dtype='float16')
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i in np.arange(len(list_IDs_temp)):
            # print(i)
            # print('i in loop for data generation: ', i+1, ' out of ', str(self.batch_size))
            ID = list_IDs_temp[i]
            # print(type(ID), ID)
            if not 'bl' in ID and not '_' in ID:  # Hammers original data
                ID = str(int(ID) + 1).zfill(2)
                path = BL_path
                # print(path)
                X_i = nib.load("".join(path + '/a' + ID + '.nii.gz'))
                Y_i = nib.load("".join(path + '/a' + ID + '-seg.nii.gz'))
            if '_' in ID and not 'bl' in ID:  # Hammers Augmented
                path = DA_path
                X_i = nib.load("".join(path + '/a' + ID + '.nii.gz'))
                Y_i = nib.load("".join(path + '/a' + ID + '-seg.nii.gz'))
            if 'bl' in ID:  # ADNI data
                path = DA_path
                # print(path)
                X_i = nib.load("".join(path + '/' + ID + 'cropbrain.nii.gz'))
                Y_i = nib.load("".join(path + '/' + ID + 'cropseg.nii.gz'))
            X_i = np.asarray(X_i.get_data())
            Y_i = np.asarray(Y_i.get_data())
            X_i = np.expand_dims(X_i, axis=4)
            # Store data in format: [batch, x, y, z, 1/classes]
            X[i,] = X_i
            Y_i = areas_to_lobes(Y_i)
            # print('areas in Y_i', np.unique(Y_i))
            Y_temp = encode1hot(Y_i)  # np.expand_dims(Y_i, axis=4)
            Y[i,] = Y_temp
        return X, Y


    def lobe_generate_validation(self, part_index, path):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(part_index)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            if len(indexes) % self.batch_size != 0:
                print(len(indexes) % self.batch_size, 'item(s) are not used for generation!')
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [part_index[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                if self.verbose and len(indexes) < 10:
                    print(i+1, '/', imax, 'list of temporary validation IDs: ', list_IDs_temp)
                if self.verbose and len(indexes) > 10:
                    print(i + 1, '/', imax, 'list of temporary training IDs: ', list_IDs_temp)
                # Generate data
                x, y = self.__lobe_validation_generation(list_IDs_temp, path)

                yield x, y



    def __lobe_validation_generation(self, list_IDs_temp, path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1)).astype(dtype='float16') # , dtype=float
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int).astype(dtype='uint8') #  #1

        # Generate validation lobe data
        for i in np.arange(len(list_IDs_temp)):
            ID = str(list_IDs_temp[i]+1)
            ID = ID.zfill(2)
            X_i = nib.load("".join(path + '/a' + ID + '.nii.gz'))
            X_i = np.asarray(X_i.get_data())
            X_i = np.expand_dims(X_i, axis=4)
            Y_i = nib.load("".join(path + '/a' + ID + '-seg.nii.gz'))
            Y_i = np.asarray(Y_i.get_data())
            Y_i = areas_to_lobes(Y_i)
            X[i, ] = X_i
            Y_i = encode1hot(Y_i)
            Y[i, ] = Y_i
        return X, Y