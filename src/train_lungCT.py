import numpy as np
import configparser
import os
from itertools import izip
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

import sys
sys.path.insert(0, './lib/')
from help_functions import *
from pre_processing import *
from extract_patches import get_data_training
from dataset import *

sys.path.insert(0, './')
from prepare_datasets import get_datasets
from models import get_fcn_model, get_patches_unet5, get_patches_unet3, get_patches_unet4, get_full_unet5, get_full_unet3
from generator import DataGenerator


# -------- Load settings from Config file -------------------------------------
config = configparser.RawConfigParser()
config.read('configuration.txt')
# Path to train images
original_imgs_train = './Lung_CT/train/all_images/'
groundTruth_imgs_train = './Lung_CT/train/all_ground_truth/'
# Path to validation images
original_imgs_val = './Lung_CT/validation/images/'
groundTruth_imgs_val = './Lung_CT/validation/ground_truth/'
# Path to the hdf5 datasets
path_data = config.get('data paths', 'path_local')
train_imgs_path = config.get('data paths', 'train_imgs_original')
train_gtruths_path = config.get('data paths', 'train_groundTruth')
# Experiment name
name_experiment = config.get('experiment name', 'name')
save_path = './' + name_experiment + '/' + name_experiment
# Training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
total_num_images = int(config.get('training settings', 'total_num_images_to_train'))

# Decide to use patches or full images as input
use_patches = config.getboolean('data attributes', 'use_patches')

# Patch data
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

img_channel = int(config.get('image attributes', 'channels'))
img_height = int(config.get('image attributes', 'height'))
img_width = int(config.get('image attributes', 'width'))
# -----------------------------------------------------------------------------


# -------- Construct and save the model arcitecture ---------------------------
if use_patches:
	model = get_patches_unet4(1, patch_height, patch_width)
else:
	model = get_patches_unet4(1, img_height, img_width)
# model.summary()
print "Final output of the network: " + str(model.output_shape)
plot_model(model, show_shapes=True, to_file=save_path + '_model.png')
json_string = model.to_json()
open(save_path + '_architecture.json', 'w').write(json_string)


checkpointer = [
	ModelCheckpoint(filepath=save_path + '_best_weights.h5',
		verbose=1,
		monitor='val_loss',
		mode='auto',
		save_best_only=True), #save at each epoch if the validation decreased
	EarlyStopping(monitor='val_loss',
		patience=10,
		verbose=1),
	ReduceLROnPlateau(monitor='val_loss',
		factor=0.5,
		patience=3,
		verbose=1,
		mode='auto',
		min_lr=1e-5)#,
	# TensorBoard(log_dir='./tensorboard_logs',
	#     histogram_freq=1,
	#     write_graph=True,
	#     write_grads=False,
	#     batch_size=batch_size,
	#     write_images=True,
	#     embeddings_freq=False,
	#     embeddings_layer_names=None,
	#     embeddings_metadata=None)
]

# -------- Custom generator class --------------------------------------------
train_generator = DataGenerator(imgs_path=original_imgs_train,
								gTruth_path=groundTruth_imgs_train,
								batch_size=batch_size, height=img_height,
								width=img_width, channels=img_channel,
								shuffle=True, name='train')
val_generator = DataGenerator(imgs_path=original_imgs_val,
								gTruth_path=groundTruth_imgs_val,
								batch_size=batch_size, height=img_height,
								width=img_width, channels=img_channel,
								shuffle=True, name='val')

print "Total number of samples to yield: " + str(len(train_generator))
print "Total number of samples to yield: " + str(len(val_generator))

history = model.fit_generator(
		generator=train_generator,
		steps_per_epoch=len(train_generator),
		epochs=N_epochs,
		verbose=2,
		callbacks=checkpointer,
		validation_data=val_generator,
		validation_steps=len(val_generator),
		max_queue_size=16,
		workers=8,	# number of cpu cores
		use_multiprocessing=True,
		shuffle=False
)

model.save_weights(save_path + '_last_weights.h5', overwrite=True)

# ------------------------------------------------------------------------------------------------------


# -------- Fit -----------------------------------------------------------------------------------------

# Get nparrays of entire images
# patches_imgs_train, patches_groundTruths_train = get_datasets(total_num_images,original_imgs_train,groundTruth_imgs_train,train_test='train')


# Extract patches from data
# patches_imgs_train, patches_groundTruths_train = get_data_training(
# 	train_imgs_original = path_data + train_imgs_path,# patches_imgs_train,
# 	train_groundTruth = path_data + train_gtruths_path, #patches_groundTruths_train,
# 	patch_height = height,
# 	patch_width = width,
# 	N_subimgs = int(config.get('training settings', 'N_subimgs')),
# 	inside_FOV = config.getboolean('training settings', 'inside_FOV'),
# 	patches = False
# )
#
# # reduce memory consumption & match output of model
# patches_groundTruths_train = masks_Unet(patches_groundTruths_train)
#
# print "PATCHES GTRUTHS TRAIN SHAPE:" + str(patches_groundTruths_train.shape)
#
# history = model.fit(patches_imgs_train, patches_groundTruths_train, epochs=50, batch_size=batch_size,
# 					verbose=2, shuffle=True, validation_split=0.2, callbacks=checkpointer)
#
#
# model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
# ------------------------------------------------------------------------------------------------------


# -------- Fit loop ----------------------------------------------------------
# # Store paths of hdf5 files
# train_hdf5_image_paths = []
# train_hdf5_groundTruth_paths = []
# # Get paths of hdf5 files
# for path, subdirs, files in os.walk(path_data):
# 	natural_sort(files)
# 	for hdf5_file in files:
# 		if "Lung_CT_datasets_imgs_train_" in hdf5_file:
# 			train_hdf5_image_paths.append(path + hdf5_file)
# 		if "Lung_CT_datasets_groundTruth_train_" in hdf5_file:
# 			train_hdf5_groundTruth_paths.append(path + hdf5_file)
#
# training_paths = zip(train_hdf5_image_paths, train_hdf5_groundTruth_paths)
#
# for e in range(50):
# 	sample_num = 0
# 	print("\n\n\n EPOCH %d of %d \n\n\n" % (e+1 ,5))
# 	for images, groundTruths in training_paths:
#
# 		# Extract patches from data
# 		patches_imgs_train, patches_groundTruths_train = get_data_training(
# 			train_imgs_original = images,
# 			train_groudTruth = groundTruths,
# 			patch_height = int(config.get('data attributes', 'patch_height')),
# 			patch_width = int(config.get('data attributes', 'patch_width')),
# 			N_subimgs = int(config.get('training settings', 'N_subimgs')),
# 			inside_FOV = config.getboolean('training settings', 'inside_FOV'),
# 			patches = False
# 		)
#
# 		print("Training on sample " + str(sample_num+1) + "/" + str(len(training_paths)))
# 		sample_num += 1
# 		history = model.fit(patches_imgs_train, patches_groundTruths_train,
#							epochs=1, batch_size=batch_size, verbose=2,
#							shuffle=True, validation_split=0.2, callbacks=checkpointer)

# model.save_weights('./'+name_experiment+'/'+name_experiment +'_last_weights.h5', overwrite=True)
# ------------------------------------------------------------------------------------------------------






# -------------------------------- Using ImageDataGenerator ----------------------------------------------
# # Training/validation data generator
# img_kwargs = dict(
# 		featurewise_center=False,
# 		samplewise_center=False,
# 		featurewise_std_normalization=False,
# 		samplewise_std_normalization=False,
#         rotation_range=0,
#         zoom_range=0.0,
#         width_shift_range=0.0,
#         height_shift_range=0.0,
#         horizontal_flip=False,
#         vertical_flip=False,
# 		data_format='channels_first'
# )
#
# gTruth_kwargs = dict(
# 		featurewise_center=False,
# 		samplewise_center=False,
# 		featurewise_std_normalization=False,
# 		samplewise_std_normalization=False,
#         rotation_range=0,
#         zoom_range=0.0,
#         width_shift_range=0.0,
#         height_shift_range=0.0,
#         horizontal_flip=False,
#         vertical_flip=False,
# 		#rescale=1./255,
# 		data_format='channels_first'
# )
#
# # Split training and validation set
# val_ratio = 0.2
# cut = int(np.ceil(len(patches_imgs_train) * val_ratio))
#
#
# # shuffle training dataset
# rng_state = np.random.get_state()
# np.random.shuffle(patches_imgs_train)
# np.random.set_state(rng_state)
# np.random.shuffle(patches_groundTruths_train)
#
#
# # split validation set
# val_images = patches_imgs_train[:cut]
# val_groundTruths = patches_groundTruths_train[:cut]
# # split training set
# train_images = patches_imgs_train[cut:]
# train_groundTruths = patches_groundTruths_train[cut:]
#
# # Instantiate image generators
# train_image_datagen = ImageDataGenerator(**img_kwargs)
# train_groundTruth_datagen = ImageDataGenerator(**gTruth_kwargs)
# val_image_datagen = ImageDataGenerator(**img_kwargs)
# val_groundTruth_datagen = ImageDataGenerator(**gTruth_kwargs)
#
# seed = 1234
# np.random.seed(seed)

# # Generator for reading train image data from folder
# train_image_generator = train_image_datagen.flow_from_directory(
# 	'./Lung_CT/train/train_images',				# contains folders 1-18
# 	target_size=(512, 512),
# 	color_mode='rgb',
# 	class_mode=None,
# 	batch_size=batch_size,
# 	shuffle=True,
# 	seed=seed
# )
#
# # Generator for reading train ground truths data from folder
# train_groundTruth_generator = train_groundTruth_datagen.flow_from_directory(
# 	'./Lung_CT/train/train_ground_truth',		# contains folders 1-18
# 	target_size=(512, 512),
# 	color_mode='grayscale',
# 	class_mode=None,
# 	batch_size=batch_size,
# 	shuffle=True,
# 	seed=seed
# )
#
# # Generator for reading validation image data from folder
# val_image_generator = val_image_datagen.flow_from_directory(
# 	'./Lung_CT/validation/val_images',		# contains folder 19
# 	target_size=(512, 512),
# 	color_mode='rgb',
# 	class_mode=None,
# 	batch_size=batch_size,
# 	shuffle=True,
# 	seed=seed
# )
#
# # Generator for reading validation ground truths data from folder
# val_groundTruth_generator = val_groundTruth_datagen.flow_from_directory(
# 	'./Lung_CT/validation/val_ground_truth',	# contains folder 19
# 	target_size=(512, 512),
# 	color_mode='grayscale',
# 	class_mode=None,
# 	batch_size=batch_size,
# 	shuffle=True,
# 	seed=seed
# )


# # Train generator
# train_image_generator = train_image_datagen.flow(train_images,
# 						shuffle=False, batch_size=batch_size, seed=seed)
# train_groundTruth_generator = train_groundTruth_datagen.flow(train_groundTruths,
# 						shuffle=False, batch_size=batch_size, seed=seed)
# train_generator = zip(train_image_generator, train_groundTruth_generator)	# izip
#
# # Validation generator
# val_image_generator = val_image_datagen.flow(val_images, shuffle=False,
#                                 batch_size=batch_size, seed=seed)
# val_groundTruth_generator = val_groundTruth_datagen.flow(val_groundTruths, shuffle=False,
#                                 batch_size=batch_size, seed=seed)
# val_generator = zip(val_image_generator, val_groundTruth_generator)			# izip
#
# history = model.fit_generator(
#         generator=train_generator,
#         steps_per_epoch=train_images.shape[0]/batch_size,
#         epochs=N_epochs,
#         verbose=2,
#         callbacks=checkpointer,
#         validation_data=val_generator,
#         validation_steps=val_images.shape[0]/batch_size,
#         shuffle=False
# )
# -------------------------------------------------------------------------------------------------------
