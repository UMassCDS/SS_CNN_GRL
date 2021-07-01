import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TrainingGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, ML_EXP, NPY_FOLDER, file_master, 
                 batch_size, dim=(128,64), n_channels=1,
                 shuffle=True, data_dir=""):

        self.filenames = np.loadtxt(data_dir + ML_EXP + file_master, dtype=str)
        print(len(self.filenames))
        self.data_dir = data_dir
        self.ML_EXP = ML_EXP
        self.NPY_FOLDER = NPY_FOLDER
        self.labels = np.array([float(f.split('/')[-1][0:4]) for f in self.filenames])
        print(len(self.labels))
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.datagen = ImageDataGenerator(
            zoom_range = 0.1,
            width_shift_range=0.2,
            height_shift_range=0.2,
            vertical_flip = True,
            horizontal_flip = True
        )

        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of of files from these indexes
        filenames = self.filenames[indexes]
        # Generate data
        X = self.__data_generation(filenames)
        y = self.labels[indexes]
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, filename in enumerate(filenames):
            image_full = np.load(self.data_dir + self.NPY_FOLDER + filename)
            image = image_full[:,:,1]
            image = self.datagen.random_transform(image.reshape([*image.shape, 1]))
            X[i,] = image
        return X

class EvalTestGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, ML_EXP , NPY_FOLDER, file_master, 
                 batch_size, dim=(128,64), n_channels=1,
                 shuffle=True, data_dir=""):    
  
        self.filenames = np.loadtxt(data_dir + ML_EXP + file_master, dtype=str)
        print(len(self.filenames))
        self.data_dir = data_dir
        self.ML_EXP = ML_EXP
        self.NPY_FOLDER = NPY_FOLDER
        self.labels = np.array([float(f.split('/')[-1][0:4]) for f in self.filenames])
        print(len(self.labels))
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of of files from these indexes
        filenames = self.filenames[indexes]
        # Generate data
        X = self.__data_generation(filenames)
        y = self.labels[indexes]
        return X, y     
    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            print('shuffling')
            np.random.shuffle(self.indexes)

    def __data_generation(self, filenames):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, filename in enumerate(filenames):
            image_full = np.load(self.data_dir + self.NPY_FOLDER + filename)
            image = image_full[:,:,1]
            X[i,] = image.reshape([*image.shape, 1])
        return X        