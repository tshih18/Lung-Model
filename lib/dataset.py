from __future__ import division, print_function

import os
import glob
import numpy as np
from math import ceil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from keras import utils
from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator
from extract_patches import get_data_training
import configparser
import gc
import sys
sys.path.insert(0, './lib/')
from help_functions import *


# def load_images(data_dir, mask='both'):
#     """Load all patient images and contours from TrainingSet, Test1Set or
#     Test2Set directory. The directories and images are read in sorted order.

#     Arguments:
#       data_dir - path to data directory (TrainingSet, Test1Set or Test2Set)

#     Output:
#       tuples of (images, masks), both of which are 4-d tensors of shape
#       (batchsize, height, width, channels). Images is uint16 and masks are
#       uint8 with values 0 or 1.
#     """
#     assert mask in ['inner', 'outer', 'both']

#     glob_search = os.path.join(data_dir, "patient*")
#     patient_dirs = sorted(glob.glob(glob_search))
#     if len(patient_dirs) == 0:
#         raise Exception("No patient directors found in {}".format(data_dir))

#     # load all images into memory (dataset is small)
#     images = []
#     inner_masks = []
#     outer_masks = []
#     for patient_dir in patient_dirs:
#         p = patient.PatientData(patient_dir)
#         images += p.images
#         inner_masks += p.endocardium_masks
#         outer_masks += p.epicardium_masks

#     # reshape to account for channel dimension
#     images = np.asarray(images)[:,:,:,None]
#     if mask == 'inner':
#         masks = np.asarray(inner_masks)
#     elif mask == 'outer':
#         masks = np.asarray(outer_masks)
#     elif mask == 'both':
#         # mask = 2 for endocardium, 1 for cardiac wall, 0 elsewhere
#         masks = np.asarray(inner_masks) + np.asarray(outer_masks)

#     # one-hot encode masks
#     dims = masks.shape
#     classes = len(set(masks[0].flatten())) # get num classes from first image
#     new_shape = dims + (classes,)
#     masks = utils.to_categorical(masks).reshape(new_shape)

#     return images, masks

def random_elastic_deformation(image, alpha, sigma, mode='nearest',
							   random_state=None):
	"""Elastic deformation of images as described in [Simard2003]_.
	.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
	   Convolutional Neural Networks applied to Visual Document Analysis", in
	   Proc. of the International Conference on Document Analysis and
	   Recognition, 2003.
	"""
	assert len(image.shape) == 3

	if random_state is None:
		random_state = np.random.RandomState(None)

	height, width, channels = image.shape

	dx = gaussian_filter(2*random_state.rand(height, width) - 1,
						 sigma, mode="constant", cval=0) * alpha
	dy = gaussian_filter(2*random_state.rand(height, width) - 1,
						 sigma, mode="constant", cval=0) * alpha

	x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
	indices = (np.repeat(np.ravel(x+dx), channels),
			   np.repeat(np.ravel(y+dy), channels),
			   np.tile(np.arange(channels), height*width))
	
	values = map_coordinates(image, indices, order=1, mode=mode)

	return values.reshape((height, width, channels))

def normalize(x, epsilon=1e-7, axis=(1,2)):
	x -= np.mean(x, axis=axis, keepdims=True)
	x /= np.std(x, axis=axis, keepdims=True) + epsilon

class Iterator(object):
	def __init__(self, training_paths,normalize, batch_size,
				 shuffle=True,
				 rotation_range=180,
				 width_shift_range=0.1,
				 height_shift_range=0.1,
				 shear_range=0.1,
				 zoom_range=0.01,
				 fill_mode='nearest',
				 alpha=500,
				 sigma=20):
		self.normalize = normalize
		self.training_paths = training_paths
		self.images = []
		self.masks = []
		self.batch_size = batch_size # len patch size
		self.count = 0
		self.shuffle = shuffle
		augment_options = {
			'rotation_range': rotation_range,
			'width_shift_range': width_shift_range,
			'height_shift_range': height_shift_range,
			'shear_range': shear_range,
			'zoom_range': zoom_range,
			'fill_mode': fill_mode,
		}
		self.idg = ImageDataGenerator(**augment_options)
		self.alpha = alpha
		self.sigma = sigma
		self.fill_mode = fill_mode
		self.i = 0
		self.index = np.arange(self.batch_size)

		self.config = configparser.RawConfigParser()
		self.config.read('configuration.txt')

		if shuffle:
			np.random.shuffle(self.index)

	def __next__(self):
		if self.count < len(self.training_paths):
			return self.next()
		return "No more images to go through"

	def next(self):
		# compute how many images to output in this batch
		# start = self.i
		gc.collect()
		self.images = []
		self.masks = []
		self.images, self.masks = get_data_training(
			DRIVE_train_imgs_original = self.training_paths[self.count][0],
			DRIVE_train_groudTruth = self.training_paths[self.count][1],  #masks
			patch_height = int(self.config.get('data attributes', 'patch_height')),
			patch_width = int(self.config.get('data attributes', 'patch_width')),
			N_subimgs = int(self.config.get('training settings', 'N_subimgs')),
			inside_FOV = self.config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
		)

			# maybe normalize image
		if self.normalize:
			normalize(self.images, axis=(1,2))

		# end = min(start + self.batch_size, len(self.images))

		self.count += 1

		# augmented_images = []
		# augmented_masks = []
		# for n in self.index: #[start:end]:
		# 	image = self.images[n]
		# 	mask = self.masks[n]

		# 	_, _, channels = image.shape

		# 	# stack image + mask together to simultaneously augment
		# 	stacked = np.concatenate((image, mask), axis=2)

		# 	# apply simple affine transforms first using Keras
		# 	augmented = self.idg.random_transform(stacked)

		# 	# maybe apply elastic deformation
		# 	if self.alpha != 0 and self.sigma != 0:
		# 		augmented = random_elastic_deformation(
		# 			augmented, self.alpha, self.sigma, self.fill_mode)

		# 	# split image and mask back apart
		# 	augmented_image = augmented[:,:,:channels]
		# 	augmented_images.append(augmented_image)
		# 	augmented_mask = np.round(augmented[:,:,channels:])
		# 	augmented_masks.append(augmented_mask)

		# self.i += self.batch_size
		# if self.i >= len(self.batch_size):
		# 	self.i = 0
		# 	if self.shuffle:
		# 		np.random.shuffle(self.index)

		self.i = 0

		if self.shuffle:
			np.random.shuffle(self.index)
		print("NEXT SAMPLE")
		# return np.asarray(augmented_images), np.asarray(augmented_masks)

		self.masks = masks_Unet(self.masks)

		return np.asarray(self.images), np.asarray(self.masks)


def create_generators(training_paths, batch_size, validation_split=0.1, mask='both',
					  shuffle_train_val=True, shuffle=True, seed=None,
					  normalize_images=True, augment_training=True,
					  augment_validation=True, augmentation_args={}):

	# images, masks = load_images(data_dir, mask)

	# before: type(masks) = uint8 and type(images) = uint16
	# convert images to double-precision
	# images = images.astype('float64')

	# # maybe normalize image
	# if normalize_images:
	# 	normalize(images, axis=(1,2))

	if seed is not None:
		np.random.seed(seed)

	# if shuffle_train_val:
	# 	# shuffle images and masks in parallel
	# 	rng_state = np.random.get_state()
	# 	np.random.shuffle(images)
	# 	np.random.set_state(rng_state)
	# 	np.random.shuffle(masks)

	# split out last %(validation_split) of images as validation set
	split_index = int((1-validation_split) * batch_size)

	if augment_training:
		train_generator = Iterator(training_paths,normalize, batch_size, shuffle=shuffle, **augmentation_args)
	else:
		idg = ImageDataGenerator()
		train_generator = idg.flow(training_paths,normalize, batch_size=batch_size, shuffle=shuffle)

	# train_steps_per_epoch = ceil(split_index / batch_size)

	if validation_split > 0.0:
		if augment_validation:
			val_generator = Iterator(training_paths,normalize, batch_size, shuffle=shuffle, **augmentation_args)
		else:
			idg = ImageDataGenerator()
			val_generator = idg.flow(training_paths,normalize, batch_size=batch_size, shuffle=shuffle)
	else:
		val_generator = None

	# val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

	# return (train_generator, train_steps_per_epoch,
	# 		val_generator, val_steps_per_epoch)

	return (train_generator, val_generator)
