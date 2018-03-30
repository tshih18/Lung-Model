import numpy as np
from PIL import Image
from prepare_datasets import natural_sort
from keras.utils import Sequence
import sys

sys.path.insert(0, './lib/')
from help_functions import *
from pre_processing import *

class DataGenerator(Sequence):
    # Generates data for Keras
    def __init__(self, imgs_path, gTruth_path, batch_size=32, height=512, width=512,
                 channels=3, shuffle=True, name=''):
        self.imgs_path = imgs_path
        self.gTruth_path = gTruth_path
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.shuffle = shuffle
        self.name = name

        self.indices_list = []
        self.img_folders = []
        self.img_file_paths = []
        self.gTruth_folders = []
        self.gTruth_file_paths = []

        # Get the full paths for training images
        for folder in os.listdir(self.imgs_path):
            self.img_folders.append(folder)
        natural_sort(self.img_folders)

        for folder in self.img_folders:
            for path, subdirs, files in os.walk(self.imgs_path + folder):
                natural_sort(files)
                for i in range(len(files)):
                    self.img_file_paths.append(self.imgs_path + folder + '/' + files[i])

        # Get the full paths for training ground truth
        for folder in os.listdir(self.gTruth_path):
            self.gTruth_folders.append(folder)
        natural_sort(self.gTruth_folders)

        for folder in self.gTruth_folders:
            for path, subdirs, files in os.walk(self.gTruth_path + folder):
                natural_sort(files)
                for i in range(len(files)):
                    self.gTruth_file_paths.append(self.gTruth_path + folder + '/' + files[i])

        # zip the train
        self.img_gTruth_paths = zip(self.img_file_paths, self.gTruth_file_paths)

        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.img_gTruth_paths) / self.batch_size))

    # Generate one batch of data
    def __getitem__(self, index):
        print "Getting batch num " + str(index) + " for " + self.name
        # Generate indexes of the batch
        indices_batch = self.indices_list[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        images, groundTruths = self.__data_generation(indices_batch)


        return images, groundTruths

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indices_list = np.arange(len(self.img_gTruth_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indices_list)

    def __data_generation(self, indices_batch):

        # Generates data containing batch_size samples
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

        images = my_PreProc(images)
        groundTruths = groundTruths/255.

        return images, groundTruths

        # # Normalize data
        # images = self.rgb2gray(images)
        # images = self.dataset_normalized(images)
        # images = images/255
        # groundTruths = groundTruths/255



    #convert RGB image in black and white
    def rgb2gray(self, rgb):
        assert (len(rgb.shape)==4)  #4D arrays
        assert (rgb.shape[1]==3)
        bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
        bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
        return bn_imgs

    def dataset_normalized(self, imgs):
        assert (len(imgs.shape)==4)  #4D arrays
        assert (imgs.shape[1]==1)  #check the channel is 1
        imgs_normalized = np.empty(imgs.shape)
        imgs_std = np.std(imgs)
        imgs_mean = np.mean(imgs)
        imgs_normalized = (imgs-imgs_mean)/imgs_std
        for i in range(imgs.shape[0]):
            imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
        return imgs_normalized
