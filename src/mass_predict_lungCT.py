###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
#Keras
from keras.models import model_from_json
from keras.models import Model

from PIL import Image
#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
sys.path.insert(0, './')
from prepare_datasets import get_datasets
# pre_processing.py
from pre_processing import my_PreProc
import tensorflow as tf
import time
import copy
import os
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -------- Load settings from Config file -------------------------------------
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')

#original test images (for FOV selection)
img_height = int(config.get('image attributes', 'height'))
img_width = int(config.get('image attributes', 'width'))
img_channel = int(config.get('image attributes', 'channels'))

use_patches = config.getboolean('data attributes', 'use_patches')
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

best_last = config.get('testing settings', 'best_last')
predict_batch = int(config.get('testing settings', 'test_batch'))

#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'

# test imgs path
original_imgs_test = "./Lung_CT/test/images/"
groundTruth_imgs_test = "./Lung_CT/test/ground_truth/"
# -----------------------------------------------------------------------------

# Get the full paths for testing images and ground truths
img_file_paths = get_image_paths(original_imgs_test)
gTruth_file_paths = get_image_paths(groundTruth_imgs_test)
img_gTruth_paths = zip(img_file_paths, gTruth_file_paths)


# Load the saved model
model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')


# Tracks and names files
file_num = 0
# Handles counting number of images per batch
count = 0

# Main loop for predictions
for i, (img_path, gTruth_path) in enumerate(img_gTruth_paths, 1):
    # Reset image and groundTruth variables after every batch
    if count % predict_batch is 0:
        image = np.empty((predict_batch, img_height, img_width, img_channel))
        groundTruth = np.empty((predict_batch, img_height, img_width))

    # Get test image
    img = Image.open(img_path)
    np_img = np.asarray(img)[:,:,:3]
    image[count] = np_img

    # Get test ground truth
    gTruth = Image.open(gTruth_path).convert("L")
    np_gTruth = np.asarray(gTruth)
    groundTruth[count] = np_gTruth

    # Go here to continue adding images
    if (count+1) < predict_batch:
        # if its not the last image
        if i != len(img_gTruth_paths):
            count += 1
            continue
        # Handle last batch of images (might not be multiple of batch size)
        else:
            # I want to truncate shape[0] img and groundTruth array
            empty_range = range(count+1, predict_batch)
            image = np.delete(image, empty_range, 0)
            groundTruth = np.delete(groundTruth, empty_range, 0)
            predict_batch = image.shape[0]

    count = 0

    # Transpose arrays for model
    image = np.transpose(image,(0,3,1,2))
    assert(image.shape == (predict_batch, img_channel, img_height, img_width))

    groundTruth = np.reshape(groundTruth,(predict_batch, 1, img_height, img_width))
    assert(groundTruth.shape == (predict_batch, 1, img_height, img_width))

    # Preprocess image
    image = my_PreProc(image)
    groundTruth = groundTruth/255.

    #Calculate the predictions
    time_start = time.time()
    print "Predicting images"
    predictions = model.predict(image, batch_size=32, verbose=2)
    print "Total prediction time: " + str(time.time() - time_start) + "seconds"
    # print "predicted images size: " + str(predictions.shape)

    if use_patches:
        predictions = pred_to_imgs(predictions, patch_height, patch_width, "original")
    else:
        predictions = pred_to_imgs(predictions, img_height, img_width, "original")
    # Transpose arrays to save as image
    predictions = np.transpose(predictions, (0,2,3,1))
    image = np.transpose(image, (0,2,3,1))
    groundTruth = np.transpose(groundTruth, (0,2,3,1))

    # Convert the prediction arrays in corresponding images
    for i in range(predict_batch):
        # Save them together as images
        threshold_confusion = 0.1
        thresh_pred = copy.deepcopy(predictions[i])
        thresh_pred[thresh_pred >= threshold_confusion] = 1 # threshold value from confusion matrix testing below
        thresh_pred[thresh_pred < threshold_confusion] = 0 # threshold value from confusion matrix testing below
        # print "Original Image Shape ", original.shape
        # print "Ground Truth Shape ", ground.shape
        # print "Prediction Shape ", prediction.shape
        # print "THRESH_PRED", thresh_pred.shape
        total_img = np.concatenate((image[i], groundTruth[i], predictions[i], thresh_pred), axis=0)
        #total_img = np.concatenate((orig_stripe, pred_stripe, thresh_pred),axis=0)
        # total_img = thresh_pred
        save_path = path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(file_num)
        # visualize(thresh_pred, save_path)
        visualize(total_img, save_path)#.show()
        print "Saving image " + save_path
        file_num += 1



# -----------------------------------------------------------------------------

#
# #====== Evaluate the results
# print "\n\n========  Evaluate the results ======================="
# #predictions only inside the FOV
# y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks)  #returns data only inside the FOV
# print "Calculating results only inside the FOV:"
# print "y scores pixels: " +str(y_scores.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]) +" (584*565==329960)"
# print "y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)"
#
# #Area under the ROC curve
# fpr, tpr, thresholds = roc_curve((y_true), y_scores)
# AUC_ROC = roc_auc_score(y_true, y_scores)
# # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
# print "\nArea under the ROC curve: " +str(AUC_ROC)
# roc_curve =plt.figure()
# plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
# plt.title('ROC curve')
# plt.xlabel("FPR (False Positive Rate)")
# plt.ylabel("TPR (True Positive Rate)")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment+"ROC.png")
#
# #Precision-recall curve
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
# precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
# recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
# AUC_prec_rec = np.trapz(precision,recall)
# print "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
# prec_rec_curve = plt.figure()
# plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
# plt.title('Precision - Recall curve')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend(loc="lower right")
# plt.savefig(path_experiment+"Precision_recall.png")
#
# plot_precision_recall_vs_threshold(precision, recall, thresholds, path_experiment)
#
# #Confusion matrix
# # this is purely for testing and the threshold value is being used for final thresholding params
#
# print "\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion)
# y_pred = np.empty((y_scores.shape[0]))
# for i in range(y_scores.shape[0]):
#   if y_scores[i]>=threshold_confusion:
#       y_pred[i]=1
#   else:
#       y_pred[i]=0
# confusion = confusion_matrix(y_true, y_pred)
# print confusion
# accuracy = 0
# if float(np.sum(confusion))!=0:
#   accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
# print "Global Accuracy: " +str(accuracy)
# specificity = 0
# if float(confusion[0,0]+confusion[0,1])!=0:
#   specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
# print "Specificity: " +str(specificity)
# sensitivity = 0
# if float(confusion[1,1]+confusion[1,0])!=0:
#   sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
# print "Sensitivity(recall): " +str(sensitivity)
# precision = 0
# if float(confusion[1,1]+confusion[0,1])!=0:
#   precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
# print "Precision: " +str(precision)
#
# #Jaccard similarity index
# jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
# print "\nJaccard similarity score: " +str(jaccard_index)
#
# #F1 score
# F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
# print "\nF1 score (F-measure): " +str(F1_score)
#
# #Save the results
# file_perf = open(path_experiment+'performances.txt', 'w')
# file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
#               + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
#               + "\nJaccard similarity score: " +str(jaccard_index)
#               + "\nF1 score (F-measure): " +str(F1_score)
#               +"\n\nConfusion matrix:"
#               +str(confusion)
#               +"\nACCURACY: " +str(accuracy)
#               +"\nSENSITIVITY: " +str(sensitivity)
#               +"\nSPECIFICITY: " +str(specificity)
#               +"\nPRECISION: " +str(precision)
#               )
# file_perf.close()
