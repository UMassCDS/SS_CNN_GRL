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
        self.KEs = np.array([float(f.split('/')[-1][0:4]) for f in self.filenames])
        self.SDs = np.array([float(f.split('/')[-1][5:9]) for f in self.filenames])
        self.labels = np.array([self.KEs,self.SDs])
        print(len(self.labels[0]))
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.datagen = ImageDataGenerator(
            zoom_range = 0.05,
            width_shift_range=0.1,
            height_shift_range=0.1,
            vertical_flip = True,
            horizontal_flip = True
        )

        
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
        self.KEs = np.array([float(f.split('/')[-1][0:4]) for f in self.filenames])
        self.SDs = np.array([float(f.split('/')[-1][5:9]) for f in self.filenames])
        self.labels = np.array([self.KEs,self.SDs])
        print(len(self.labels[0]))
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
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

        for i, filename in enumerate(filenames):
            image_full = np.load(self.data_dir + self.NPY_FOLDER + filename)
            image = image_full[:,:,1]
            X[i,] = image.reshape([*image.shape, 1])
        return X        