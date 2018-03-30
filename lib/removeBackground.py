import numpy as np
import cv2
import os
from scipy.signal import savgol_filter
import sys
sys.path.insert(0, './')
from prepare_datasets import get_datasets, natural_sort

# Finds all edges in imgae
def edgedetect(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255
    return sobel

# Determines which contours to keep
def findSignificantContours(img, edgeImg):
    image, contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = edgeImg.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv2.contourArea(contour)
        if area > tooSmall:

            # Use Savitzky-Golay filter to smoothen contour. (curved edges)
            window_size = int(round(min(img.shape[0], img.shape[1]) * 0.05)) # Consider each window to be 5% of image dimensions
            x = savgol_filter(contour[:,0,0], window_size * 2 + 1, 3)
            y = savgol_filter(contour[:,0,1], window_size * 2 + 1, 3)

            approx = np.empty((x.size, 1, 2))
            approx[:,0,0] = x
            approx[:,0,1] = y
            approx = approx.astype(int)
            contour = approx

            # Contour smoothing for straight edges
            # epsilon = 0.10*cv2.arcLength(contour,True)
            # approx = cv2.approxPolyDP(contour, 3,True)
            # contour = approx

            significant.append([contour, area])

            # Draw the contour on the original image
            cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant];


def removeBackground(img_path, save_path, count):
    img = cv2.imread(img_path)

    # Slight gaussian blue to reduce noice from image
    blurImg = cv2.GaussianBlur(img, (17,17), 0)

    # Edge detection: Sobel operator
    edgeImg = np.max(np.array([edgedetect(blurImg[:,:, 0]),
                    edgedetect(blurImg[:,:, 1]),
                    edgedetect(blurImg[:,:, 2])]), axis=0)

    # Reduce noise on edge image - comment: WTF HELLA GUUD **IMPORTANT STEP
    mean = np.mean(edgeImg);
    # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg <= mean] = 0;

    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    # Find contours, returns 1 list of countor
    significant = findSignificantContours(img, edgeImg_8u)

    # Create mask
    mask = edgeImg.copy()
    mask[mask > 0] = 0

    # Make a binary mask filling region with largest contour white
    cv2.fillPoly(mask, significant, 255)

    # Invert mask
    mask = np.logical_not(mask)

    #Finally remove the background
    img[mask] = 0

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    cv2.imwrite(save_path + "VESSEL12_20-slice" + str(count) + ".png", img)

    return img


if __name__ == '__main__':
    img_path = '/home/aether/Desktop/Medical Image Segmentation/lung_unet_model/Lung_CT/test/images/VESSEL12_20_png/'
    save_path = '/home/aether/Desktop/Medical Image Segmentation/lung_unet_model/Lung_CT/test/images/VESSEL12_20_png_NoBackground/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for path, subdirs, files in os.walk(img_path):
        natural_sort(files)
        for i, image_file in enumerate(files):
            removeBackground(img_path + image_file, save_path, i)
