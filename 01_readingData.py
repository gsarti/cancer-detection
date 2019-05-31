# -*- coding: utf-8 -*-
"""
01. Reading data
Created on Wed May 29 15:17:33 2019
@author: leona
"""

# Check the working directory
import sys
sys.path


# Libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook
import math

from sklearn.model_selection import train_test_split


data = pd.read_csv('histopathologic-cancer-detection/train_labels.csv')

train_path = 'histopathologic-cancer-detection/train/'
test_path = 'histopathologic-cancer-detection/test/'


# label distributions
data.shape

count = data['label'].value_counts()
freq = data['label'].value_counts(normalize = True)
# freq = count/len(data)

y_freq = pd.DataFrame({'label': list(map(str, count.index)),
                       'count': count,
                       'freq': freq})

y_freq
y_freq.loc[:, 'freq']

plt.clf()
plt.subplot()
plt.bar(y_freq.loc[:, 'label'],
        y_freq.loc[:, 'freq'])
plt.xlabel('labels')
plt.ylabel('Relative frequency')
plt.title('Labels distributions')
plt.show()
# plt.grid(True)


# Visualize images
def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img / 255


# random sampling
shuffled_data = shuffle(data)

fig, ax = plt.subplots(2,5, figsize=(20,8))
#fig, ax = plt.subplots(4,5, figsize=(20,16))
fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20)
# Negatives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[0,i].imshow(readImage(path + '.tif'))
    # Create a Rectangle patch
    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',
                            facecolor='none', linestyle=':', capstyle='round')
    ax[0,i].add_patch(box)
ax[0,0].set_ylabel('Negative samples', size='large')
# Positives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[1,i].imshow(readImage(path + '.tif'))
    # Create a Rectangle patch
    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',
                            facecolor='none', linestyle=':', capstyle='round')
    ax[1,i].add_patch(box)
ax[1,0].set_ylabel('Tumor tissue samples', size='large')


# Do it with the first images:
# 10 positive and 10 negative

#fig, ax = plt.subplots(2,5, figsize=(20,8))
fig, ax = plt.subplots(4,5, figsize=(20,16))
fig.suptitle('Histopathologic scans of lymph node sections',fontsize=20)
# Negatives

for j in [0,1]:
    for i, idx in enumerate(data[data['label'] == 0]['id'][(j*5):((j+1)*5)]):
        path = os.path.join(train_path, idx)
        ax[j,i].imshow(readImage(path + '.tif'))
        # Create a Rectangle patch
        box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='g',
                            facecolor='none', linestyle=':', capstyle='round')
        ax[j,i].add_patch(box)
    ax[j,0].set_ylabel('Negative samples', size='large')

for j in [0,1]:
    # Positives
    for i, idx in enumerate(data[data['label'] == 1]['id'][(j*5):((j+1)*5)]):
        path = os.path.join(train_path, idx)
        ax[(j+2),i].imshow(readImage(path + '.tif'))
        # Create a Rectangle patch
        box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',
                            facecolor='none', linestyle=':', capstyle='round')
        ax[(j+2),i].add_patch(box)
    ax[(j+2),0].set_ylabel('Tumor tissue samples', size='large')



# Augmentation with OpenCV

# rotarion
# crop
# flip
# lightning
# zoom (NO)
# Gaussian blur (NO)

import random
ORIGINAL_SIZE = 96      # original size of the images - do not change

# AUGMENTATION VARIABLES
CROP_SIZE = 90          # final size after crop
#CROP_SIZE = 66
# Ha senso mettere crop = 66, così non ci sono pixel neri dopo la rotazione
RANDOM_ROTATION = 3     # range (0-180), 180 allows all rotation variations, 0=no change
#RANDOM_ROTATION = 45
RANDOM_SHIFT = 2        # center crop shift in x and y axes, 0=no change. This cannot be more than (ORIGINAL_SIZE - CROP_SIZE)//2 
RANDOM_BRIGHTNESS = 7   # range (0-100), 0=no change
#RANDOM_BRIGHTNESS = 10
RANDOM_CONTRAST = 5     # range (0-100), 0=no change
RANDOM_90_DEG_TURN = 1  # 0 or 1= random turn to left or right

def readCroppedImage(path, augmentations = True):
    # augmentations parameter is included for counting statistics from images, where we don't want augmentations
    
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    
    if(not augmentations):
        return rgb_img / 255
    
    #random rotation
    rotation = random.randint(-RANDOM_ROTATION,RANDOM_ROTATION)
#    rotation = 45
    if(RANDOM_90_DEG_TURN == 1):
        rotation += random.randint(-1,1) * 90
    M = cv2.getRotationMatrix2D((48,48),rotation,1)   # the center point is the rotation anchor
    rgb_img = cv2.warpAffine(rgb_img,M,(96,96))
    
    #random x,y-shift
    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    
    # crop to center and normalize to 0-1 range
    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
    end_crop = start_crop + CROP_SIZE
    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255
    
    # Random flip
    flip_hor = bool(random.getrandbits(1))
    flip_ver = bool(random.getrandbits(1))
    if(flip_hor):
        rgb_img = rgb_img[:, ::-1]
    if(flip_ver):
        rgb_img = rgb_img[::-1, :]
        
    # Random brightness
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
    rgb_img = rgb_img + br
    
    # Random contrast
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
    rgb_img = rgb_img * cr
    
    # clip values to 0-1 range
    rgb_img = np.clip(rgb_img, 0, 1.0)
    
    return rgb_img



# =============================================================================
# # Nota: dividere o no per 255 non cambia niente
# fig, ax = plt.subplots(2,2, figsize=(20,20))
# ax[0,0].imshow(readImage(path + '.tif'))
# ax[0,1].imshow(readCroppedImage(path + '.tif', augmentations = False))
# ax[1,0].imshow(readCroppedImage(path + '.tif', augmentations = True))
# ax[1,1].imshow(readCroppedImage(path + '.tif', augmentations = True))
#rgb_img = readImage(path + '.tif')
#plt.imshow(rgb_img)    
# =============================================================================


# View augmented images
fig, ax = plt.subplots(2,5, figsize=(20,8))
fig.suptitle('Cropped histopathologic scans of lymph node sections',fontsize=20)
# Negatives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[0,i].imshow(readCroppedImage(path + '.tif'))
ax[0,0].set_ylabel('Negative samples', size='large')
# Positives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[1,i].imshow(readCroppedImage(path + '.tif'))
ax[1,0].set_ylabel('Tumor tissue samples', size='large')


# See augmentation effect
fig, ax = plt.subplots(1,5, figsize=(30,6))
fig.suptitle('Random augmentations to the same image',fontsize=20)
# Negatives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:1]):
    path = os.path.join(train_path, idx)
    ax[0].imshow(readImage(path + '.tif'))
    ax[0].set_title('Original image')
    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',
                            facecolor='none', linestyle=':', capstyle='round')
    ax[0].add_patch(box)
    for j in range(1,5):
        ax[j].imshow(readCroppedImage(path + '.tif'))
        ax[j].set_title('Cropped image')
        box = patches.Rectangle((32-(ORIGINAL_SIZE - CROP_SIZE)/2,32-(ORIGINAL_SIZE - CROP_SIZE)/2),
                                32,32,linewidth=4,edgecolor='b',
                            facecolor='none', linestyle=':', capstyle='round')
        ax[j].add_patch(box)



# Count statistics on channels
# Search too dark or too bright images

# As we count the statistics, we can check if there are any completely black or white images
dark_th = 10 / 255      # If no pixel reaches this threshold, image is considered too dark
bright_th = 245 / 255   # If no pixel is under this threshold, image is considerd too bright
too_dark_idx = []
too_bright_idx = []

x_tot = np.zeros(3)
x2_tot = np.zeros(3)
counted_ones = 0
for i, idx in tqdm_notebook(enumerate(shuffled_data['id']), 'computing statistics...(220025 it total)'):
    path = os.path.join(train_path, idx)
    imagearray = readCroppedImage(path + '.tif', augmentations = False).reshape(-1,3)
    # is this too dark
    if(imagearray.max() < dark_th):
        too_dark_idx.append(idx)
        continue # do not include in statistics
    # is this too bright
    if(imagearray.min() > bright_th):
        too_bright_idx.append(idx)
        continue # do not include in statistics
    x_tot += imagearray.mean(axis=0)
    x2_tot += (imagearray**2).mean(axis=0)
    counted_ones += 1
    
channel_avr = x_tot/counted_ones
channel_std = np.sqrt(x2_tot/counted_ones - channel_avr**2)

channel_avr,channel_std


print('There was {0} extremely dark image'.format(len(too_dark_idx)))
print('and {0} extremely bright images'.format(len(too_bright_idx)))
print('Dark one:')
print(too_dark_idx)
print('Bright ones:')
print(too_bright_idx)


data[data['id'].isin(too_dark_idx)]
data[data['id'].isin(too_bright_idx)]


# Plot too dark and too bright images

fig, ax = plt.subplots(2,6, figsize=(25,9))
fig.suptitle('Almost completely black or white images',fontsize=20)
# Too dark
i = 0
for idx in np.asarray(too_dark_idx)[:min(6, len(too_dark_idx))]:
    lbl = shuffled_data[shuffled_data['id'] == idx]['label'].values[0]
    path = os.path.join(train_path, idx)
    print("Mean value: image " + str(i))
    print(np.mean(readCroppedImage(path + '.tif', augmentations = False)))
    ax[0,i].imshow(readCroppedImage(path + '.tif', augmentations = False))
    ax[0,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8)
    i += 1
ax[0,0].set_ylabel('Extremely dark images', size='large')
for j in range(min(6, len(too_dark_idx)), 6):
    ax[0,j].axis('off') # hide axes if there are less than 6
# Too bright
i = 0
for idx in np.asarray(too_bright_idx)[:min(6, len(too_bright_idx))]:
    lbl = shuffled_data[shuffled_data['id'] == idx]['label'].values[0]
    path = os.path.join(train_path, idx)
    print("Mean value: image " + str(i))
    print(np.mean(readCroppedImage(path + '.tif', augmentations = False)))
    ax[1,i].imshow(readCroppedImage(path + '.tif', augmentations = False))
    ax[1,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8)
    i += 1
ax[1,0].set_ylabel('Extremely bright images', size='large')
for j in range(min(6, len(too_bright_idx)), 6):
    ax[1,j].axis('off') # hide axes if there are less than 6




