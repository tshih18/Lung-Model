import numpy as np
import ConfigParser
from matplotlib import pyplot as plt
import tensorflow as tf
import time
import copy
import os
import sys
from PIL import Image
from keras.models import model_from_json, Model
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix,
                            precision_recall_curve, jaccard_similarity_score,
                            f1_score

sys.path.insert(0, './lib/')
from help_functions import *
from extract_patches import *
sys.path.insert(0, './')
from prepare_datasets import get_datasets
from pre_processing import my_PreProc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
count = 0

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, path_experiment):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    print("thresholds" + str(thresholds))
    print("precisions" + str(precisions))
    print("recall" + str(recalls))
    plt.xlim([-1, 1])
    plt.savefig(path_experiment+"Precision_vs_Recall.png")

# -------- Load settings from Config file -------------------------------------
config = ConfigParser.RawConfigParser()
config.read('configuration.txt')

# Run the training on invariant or local
path_data = config.get('data paths', 'path_local')

# Original test images (for FOV selection)
test_imgs_original_path = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(test_imgs_original_path)
test_imgs_groundTruth_path = path_data + config.get('data paths', 'test_groundTruth')

# Dimension of the images
img_height = config.get('image attributes', 'height')
img_width = config.get('image attributes', 'width')
img_channel = config.get('image attributes', 'channels')
# Dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

# the number of images to test
num_test_imgs = int(config.get('testing settings', 'total_num_images_to_test'))

#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)

# Model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' +name_experiment +'/'
best_last = config.get('testing settings', 'best_last')

# test imgs path
original_imgs_test = "./Lung_CT/test/images/"
groundTruth_imgs_test = "./Lung_CT/test/ground_truth/"

# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

# Average mode
average_mode = config.getboolean('testing settings', 'average_mode')
# -----------------------------------------------------------------------------


# -------- Predict on a folder of images --------------------------------------
# Path to test images
original_imgs_test = './Lung_CT/test/images/'
groundTruth_imgs_test = './Lung_CT/test/ground_truth/'

# Get the full paths for testing images and ground truths
img_file_paths = get_image_paths(original_imgs_test)
gTruth_file_paths = get_image_paths(groundTruth_imgs_test)
img_gTruth_paths = zip(img_file_paths, gTruth_file_paths)

images_to_predict = 1
for img_path, gTruth_path in img_gTruth_paths:
    image = np.empty((images_to_predict, img_height, img_width, img_channel))
    groundTruth = np.empty((images_to_predict, img_height, img_width))

    # Get test image
    img = Image.open(img_path)
    np_img = np.asarray(img)[:,:,:3]
    print("imgs max: " +str(np.max(img)))
    print("imgs min: " +str(np.min(img)))
    image[0] = np_img

    image = np.transpose(image,(0,3,1,2))
    assert(image.shape == (images_to_predict,img_channel,img_height,img_width))

    # Get test ground truth
    gTruth = Image.open(gTruth_path).convert("L")
    np_gTruth = np.asarray(gTruth)
    print("ground truth max: " + str(np.max(groundTruth)))
    print("ground truth min: " + str(np.min(groundTruth)))
    groundTruth[0] = np_gTruth

    groundTruth = np.reshape(groundTruth,(images_to_predict,1,img_height,img_width))
    assert(groundTruth.shape == (images_to_predict,1,img_height,img_width))

    # Preprocess image
    image = my_PreProc(image)
    groundTruth = groundTruth/255.

# -----------------------------------------------------------------------------



# -------- Load the data from hdf5 file and divide in patches -----------------
patches_imgs_test = None
new_height = None
new_width = None
test_groundTruth  = None
patches_masks_test = None

if average_mode == True:
    patches_imgs_test, new_height, new_width, test_groundTruth = get_data_testing_overlap(
    #patches_imgs_test, new_height, new_width = get_data_testing_overlap(
        test_imgs_original = test_imgs_original_path,
        test_groudTruth = test_imgs_groundTruth_path,
        Imgs_to_test = num_test_imgs,
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width,
        patches = False
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        test_imgs_original = test_imgs_original_path,
        test_groudTruth = test_imgs_groundTruth_path,
        Imgs_to_test = num_test_imgs,
        patch_height = patch_height,
        patch_width = patch_width,
    )



#================ Run the prediction of the patches ==================================
model = model_from_json(open(path_experiment + name_experiment + '_architecture.json').read())
model.load_weights(path_experiment + name_experiment + '_' + best_last + '_weights.h5')
time_start = time.time()
print "Predicting images"
predictions = model.predict(patches_imgs_test, batch_size=32, verbose=2)
print "Total prediction time:" + str(time.time() - time_start) + "seconds"
print "predicted images size :"
print predictions.shape

#===== Convert the prediction arrays in corresponding images
if use_patches:
    pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
else:
    pred_patches = pred_to_imgs(predictions, img_height, img_width, "original")
# pred_patches = predictions


# -------- Elaborate and visualize the predicted images -----------------------
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
    orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
    gtruth_masks = test_groundTruth  #ground truth masks
else:
    pred_imgs = recompone(pred_patches,13,12)       # predictions
    orig_imgs = recompone(patches_imgs_test,13,12)  # originals
    gtruth_masks = recompone(patches_masks_test,13,12)  #masks
# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
# kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
## back to original dimensions
orig_imgs = orig_imgs[:,:,0:img_height,0:img_width]
pred_imgs = pred_imgs[:,:,0:img_height,0:img_width]
gtruth_masks = gtruth_masks[:,:,0:img_height,0:img_width]
print "Orig imgs shape: " +str(orig_imgs.shape)
print "pred imgs shape: " +str(pred_imgs.shape)
print "Gtruth imgs shape: " +str(gtruth_masks.shape)
visualize(group_images(orig_imgs,N_visual),path_experiment+"all_originals")#.show()
visualize(group_images(pred_imgs,N_visual),path_experiment+"all_predictions")#.show()
visualize(group_images(gtruth_masks,N_visual),path_experiment+"all_groundTruths")#.show()

for image in pred_imgs:
    print("Max value:", np.max(image))

#visualize results comparing mask and prediction:
# assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted%group==0)

threshold_confusion = 0.1 # this is used below for evaluation

# Save predictions
for i in range(int(N_predicted/group)):
    orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
    masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
    pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)
    thresh_pred = copy.deepcopy(pred_stripe)
    thresh_pred[thresh_pred >= threshold_confusion] = 1 # threshold value from confusion matrix testing below
    thresh_pred[thresh_pred < threshold_confusion] = 0 # threshold value from confusion matrix testing below
    print "ORIG STRIPE", orig_stripe.shape
    print "MASKS_STRIPE", masks_stripe.shape
    print "PRED_STRIPE", pred_stripe.shape
    print "THRESH_PRED", thresh_pred.shape
    total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe, thresh_pred),axis=0)
    #total_img = np.concatenate((orig_stripe, pred_stripe, thresh_pred),axis=0)
    # total_img = thresh_pred
    # visualize(thresh_pred,path_experiment+name_experiment +"_Threshold_Prediction"+str(count))
    visualize(total_img,path_experiment+name_experiment +"_Original_GroundTruth_Prediction"+str(count))#.show()
    count += 1



# -------- Evaluate the results -----------------------------------------------
print "\n\n========  Evaluate the results ======================="
#predictions only inside the FOV
y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks)  #returns data only inside the FOV
print "Calculating results only inside the FOV:"
print "y scores pixels: " +str(y_scores.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(pred_imgs.shape[0]*pred_imgs.shape[2]*pred_imgs.shape[3]) +" (584*565==329960)"
print "y true pixels: " +str(y_true.shape[0]) +" (radius 270: 270*270*3.14==228906), including background around retina: " +str(gtruth_masks.shape[2]*gtruth_masks.shape[3]*gtruth_masks.shape[0])+" (584*565==329960)"

#Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print "\nArea under the ROC curve: " +str(AUC_ROC)
roc_curve =plt.figure()
plt.plot(fpr,tpr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"ROC.png")

#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path_experiment+"Precision_recall.png")

plot_precision_recall_vs_threshold(precision, recall, thresholds, path_experiment)

#Confusion matrix
# this is purely for testing and the threshold value is being used for final thresholding params

print "\nConfusion matrix:  Costum threshold (for positive) of " +str(threshold_confusion)
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    if y_scores[i]>=threshold_confusion:
        y_pred[i]=1
    else:
        y_pred[i]=0
confusion = confusion_matrix(y_true, y_pred)
print confusion
accuracy = 0
if float(np.sum(confusion))!=0:
    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
print "Global Accuracy: " +str(accuracy)
specificity = 0
if float(confusion[0,0]+confusion[0,1])!=0:
    specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
print "Specificity: " +str(specificity)
sensitivity = 0
if float(confusion[1,1]+confusion[1,0])!=0:
    sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
print "Sensitivity(recall): " +str(sensitivity)
precision = 0
if float(confusion[1,1]+confusion[0,1])!=0:
    precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
print "Precision: " +str(precision)

#Jaccard similarity index
jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
print "\nJaccard similarity score: " +str(jaccard_index)

#F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print "\nF1 score (F-measure): " +str(F1_score)

#Save the results
file_perf = open(path_experiment+'performances.txt', 'w')
file_perf.write("Area under the ROC curve: "+str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " +str(AUC_prec_rec)
                + "\nJaccard similarity score: " +str(jaccard_index)
                + "\nF1 score (F-measure): " +str(F1_score)
                +"\n\nConfusion matrix:"
                +str(confusion)
                +"\nACCURACY: " +str(accuracy)
                +"\nSENSITIVITY: " +str(sensitivity)
                +"\nSPECIFICITY: " +str(specificity)
                +"\nPRECISION: " +str(precision)
                )
file_perf.close()
