"""
Keras data generator for file, as seen in this blog post:

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

by Afshine Amidi and Shervine Amidi
"""

import numpy as np
from keras.utils import Sequence, to_categorical
from PIL import Image


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, train_dir='data/train/', batch_size=32, dim=(96, 96), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dir = train_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.shape = (batch_size, *dim, n_channels)
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(self.shape)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.array(Image.open(train_dir + ID + '.tif')

            # Store class
            y[i] = self.labels[ID]

        # Standardize images
        X = X.astype('float32') / 255

        return X, to_categorical(y, num_classes=self.n_classes)
