#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image
import cv2
import configparser
import re
from scipy.misc import imshow
import cv2

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./Lung_CT/train/images/"
groundTruth_imgs_train = "./Lung_CT/train/ground_truth/"
#test
original_imgs_test = "./Lung_CT/test/images/"
groundTruth_imgs_test = "./Lung_CT/test/ground_truth/"
#---------------------------------------------------------------------------------------------

#------------- Load settings from Config file ------------------------------------------------
config = configparser.RawConfigParser()
config.read('configuration.txt')
# Get number of images for training and testing
train_imgs = int(config.get('training settings', 'total_num_images_to_train'))
test_imgs = int(config.get('testing settings', 'total_num_images_to_test'))
# Get path to save datasets
dataset_path = config.get('data paths', 'path_local')
train_imgs_path = config.get('data paths', 'train_imgs_original')
train_gtruths_path = config.get('data paths', 'train_groundTruth')
test_imgs_path = config.get('data paths', 'test_imgs_original')
test_gtruths_path = config.get('data paths', 'test_groundTruth')

# Specificity dimensions of images
channels = 3
height = 512
width = 512

# Test if a given chunk is an int
def tryint(c):
    try:
        return int(c)
    except:
        return c

# turn a string into a list of string and number chunks
def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

# sort the given list
def natural_sort(list):
    list.sort(key=alphanum_key)

# Separates each folder into a hdf5 dataset (for a lot of data)
def mass_get_datasets(Nimgs,imgs_dir,groundTruth_dir,train_test="null"):

    folders = []
    # Get all the folder names
    for folder in os.listdir(imgs_dir):
        folders.append(folder)
    natural_sort(folders)

    hdf5_num = 0
    # Loop through image VESSELS folder
    for folder in folders:
        # Loop through files in VESSELS folder
        for path, subdirs, files in os.walk(imgs_dir + folder):
            natural_sort(files)
            # reset imgs np array after each folder
            imgs = np.empty((len(files), height, width, channels))

            # Loop through each image
            for i in range(len(files)):
                print("original image: " + files[i])
                img = Image.open(imgs_dir + folder + "/" + files[i])#.convert("L")
                np_img = np.asarray(img)[:,:,:3]
                imgs[i] = np_img

            # print image max/min values
            print("Folder " + folder + " imgs max: " +str(np.max(imgs)))
            print("Folder " + folder + " imgs min: " +str(np.min(imgs)))

            # reshaping imgs for standard tensors
            imgs = np.transpose(imgs,(0,3,1,2))
            assert(imgs.shape == (len(files),channels,height,width))

            # save each folder as an hdf5 file
            if train_test == "train":
                name = train_imgs_path[:-5] + "_" + str(hdf5_num) + ".hdf5"

            else:
                name = test_imgs_path[:-5] + "_" + str(hdf5_num) + ".hdf5"

            path = dataset_path + name
            print("Saving " + train_test + " image dataset " + str(hdf5_num+1) + "/" + str(len(folders)) +
                    " as " + name + " containing " + str(len(files)) + " images")
            write_hdf5(imgs, path)
            hdf5_num += 1

    # ---------------------------------------------------------------------------

    folders = []
    # Get all the folder names
    for folder in os.listdir(groundTruth_dir):
        folders.append(folder)
    natural_sort(folders)

    hdf5_num = 0
    # Loop through ground truths VESSELS folder
    for folder in folders:
        # Loop through files in VESSELS folder
        for path, subdirs, files in os.walk(groundTruth_dir + folder):
            natural_sort(files)
            # reset imgs np array after each folder
            groundTruth = np.empty((len(files), height, width))
            # Loop through each image
            for i in range(len(files)):
                print("ground truth image: " + files[i])
                g_truth = Image.open(groundTruth_dir + folder + "/" + files[i]).convert("L")
                np_g_truth = np.asarray(g_truth)
                groundTruth[i] = np_g_truth

            # print image max/min values
            print("Folder " + folder + " ground truth max: " + str(np.max(groundTruth)))
            print("Folder " + folder + " ground truth min: " + str(np.min(groundTruth)))

            # reshaping ground truth
            groundTruth = np.reshape(groundTruth,(len(files),1,height,width))
            assert(groundTruth.shape == (len(files),1,height,width))

            # save each folder as an hdf5 file
            if train_test == "train":
                name = train_gtruths_path[:-5] + "_" + str(hdf5_num) + ".hdf5"

            else:
                name = test_gtruths_path[:-5] + "_" + str(hdf5_num) + ".hdf5"

            path = dataset_path + name
            print("Saving " + train_test + " image dataset " + str(hdf5_num+1) + "/" + str(len(folders)) +
                    " as " + name + " containing " + str(len(files)) + " images")
            write_hdf5(groundTruth, path)
            hdf5_num += 1



def get_datasets(Nimgs,imgs_dir,groundTruth_dir,train_test="null"):
    imgs = np.empty((Nimgs, height, width, channels))
    groundTruth = np.empty((Nimgs, height, width))

    # Save original images
    for path, subdirs, files in os.walk(imgs_dir):
        natural_sort(files)
        for i in range(len(files)):
            # original image
            print("original image: " + files[i])
            img = Image.open(imgs_dir + files[i])#.convert("L")
            np_img = np.asarray(img)[:,:,:3]

            # imgs_std = np.std(np_img)
            # imgs_mean = np.mean(np_img)
            # np_img = (np_img-imgs_mean)/imgs_std

            imgs[i] = np_img

    # Save ground truth images
    for path, subdirs, files in os.walk(groundTruth_dir):
        natural_sort(files)
        for i in range(len(files)):
            # ground truth
            print("ground truth name: " + files[i])
            g_truth = Image.open(groundTruth_dir + files[i]).convert("L")
            np_g_truth = np.asarray(g_truth)

            groundTruth[i] = np_g_truth


    # print image max/min values
    print("imgs max: " +str(np.max(imgs)))
    print("imgs min: " +str(np.min(imgs)))
    print("ground truth max: " + str(np.max(groundTruth)))
    print("ground truth min: " + str(np.min(groundTruth)))

    # reshaping imgs for standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))

    # reshaping ground truth
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))

    return imgs, groundTruth

if __name__ == '__main__':
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


    # get datasets of many files separated in samples of hdf5 files
    # mass_get_datasets(train_imgs,original_imgs_train,groundTruth_imgs_train,"train")
    # mass_get_datasets(test_imgs,original_imgs_test,groundTruth_imgs_test,"test")

    #getting the training datasets
    imgs_train, groundTruth_train = get_datasets(train_imgs,original_imgs_train,groundTruth_imgs_train,"train")
    print("saving train datasets")
    write_hdf5(imgs_train, dataset_path + train_imgs_path)
    write_hdf5(groundTruth_train, dataset_path + train_gtruths_path)

    #getting the testing datasets
    imgs_test, groundTruth_test = get_datasets(test_imgs,original_imgs_test,groundTruth_imgs_test,"test")
    print("saving test datasets")
    write_hdf5(imgs_test,dataset_path + test_imgs_path)
    write_hdf5(groundTruth_test, dataset_path + test_gtruths_path)
