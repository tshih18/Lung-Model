import skimage.io as skio
import skimage as sk
import numpy as np
import os
from matplotlib import pyplot as plt

def ThreshImages(dir):
    for filename in os.listdir(dir):
        if "bone_model_Original_GroundTruth_Prediction0" in filename:
             img_arr = skio.imread(dir + filename)
             img_arr = sk.img_as_float(img_arr)

             print(np.max(img_arr))
             max_val = np.max(img_arr)
             #img_arr[img_arr > max_val*0.9] = 0
             img_arr[img_arr < max_val*0.1] = 0
             img_arr[img_arr != 0] = 1
             plt.imshow(img_arr, cmap="gray")
             plt.show()
             # skio.imsave(dir + filename[:-4] + "_thresh.png", img_arr)
             break

ThreshImages("/home/aether/Desktop/Medical Image Segmentation/bones-unet/bone_model/")
