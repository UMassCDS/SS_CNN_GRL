import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CrustGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, master_file, batch_size, group, dim=(128,64), n_channels=1,
                 data_dir=""):
   
        # Labels will be a float with values between 0-3 (with 1 decimal point)
        self.filenames = np.loadtxt(data_dir + master_file, dtype=str)
        print('sample#:',len(self.filenames))
        print(self.filenames)
        self.data_dir = data_dir
        self.group = group
        self.KEs = np.array([float(f[0:4]) for f in self.filenames])
        self.SDs = np.array([float(f[5:9]) for f in self.filenames])
        self.labels = np.array([self.KEs,self.SDs])
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = False
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels[0]) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of of files from these indexes
        filenames = self.filenames[indexes]
        # Generate data
        X = self.__data_generation(filenames)
        y = np.array(self.labels[:,indexes]).T
        return X, y  
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels[0]))
        if self.shuffle == True:
            print('shuffling')
            np.random.shuffle(self.indexes)


    def __data_generation(self, filenames):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, filename in enumerate(filenames):
            # Store sample
            image = np.load(self.data_dir +  self.group + filename)
            X[i,] = image.reshape([*image.shape, 1])
        return X