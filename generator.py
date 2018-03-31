import numpy as np
from PIL import Image
from prepare_datasets import natural_sort
from keras.utils import Sequence
import sys
import os

sys.path.insert(0, './lib/')
from help_functions import *
from pre_processing import *

# Custom generator that reads files from a folder batches at a time
# to handle memory overloading
class DataGenerator(Sequence):
    def __init__(self, imgs_path, gTruth_path, batch_size, height, width,
                 channels, shuffle, name):
        self.imgs_path = imgs_path
        self.gTruth_path = gTruth_path
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.shuffle = shuffle
        self.name = name

        self.indices_list = []

        # Get the full paths for training images and ground truths
        self.img_file_paths = get_image_paths(self.imgs_path)
        self.gTruth_file_paths = get_image_paths(self.gTruth_path)
        self.img_gTruth_paths = zip(self.img_file_paths, self.gTruth_file_paths)

        self.on_epoch_end()

    # Denotes the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.img_gTruth_paths) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        # Generate indexes of the batch
        indices_batch = self.indices_list[index*self.batch_size:(index+1)*self.batch_size]
        # print "Getting batch num " + str(index) + " for " + self.name

        # Generate data
        images, groundTruths = self.__data_generation(indices_batch)

        return images, groundTruths

    # Gets called after every epoch
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indices_list = np.arange(len(self.img_gTruth_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices_list)

    # Generates data containing batch_size samples
    def __data_generation(self, indices_batch):
        # Initialization
        images = np.empty((self.batch_size, self.height, self.width , self.channels))
        groundTruths = np.empty((self.batch_size, self.height, self.width))

        # Generate data
        for i, index in enumerate(indices_batch):
            img_file = self.img_gTruth_paths[index][0]
            img = Image.open(img_file)
            np_img = np.asarray(img)[:,:,:3]
            images[i] = np_img

            gTruth_file = self.img_gTruth_paths[index][1]
            gTruth = Image.open(gTruth_file).convert("L")
            np_gTruth = np.asarray(gTruth)
            groundTruths[i] = np_gTruth

        # Reshape data
        images = np.transpose(images,(0,3,1,2))
        assert(images.shape == (self.batch_size, self.channels, self.height, self.width))

        groundTruths = np.reshape(groundTruths,(self.batch_size, 1, self.height, self.width))
        assert(groundTruths.shape == (self.batch_size, 1, self.height, self.width))

        # Preprocess images
        images = my_PreProc(images)
        groundTruths = groundTruths/255.

        # Reshape ground truths to match output of model
        groundTruths = masks_Unet(groundTruths)

        return images, groundTruths
