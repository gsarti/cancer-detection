# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:50:46 2019

@author: leona
"""

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

# we read the csv file earlier to pandas dataframe, now we set index to id so we can perform
train_df = data.set_index('id')

#If removing outliers, uncomment the four lines below
#print('Before removing outliers we had {0} training samples.'.format(train_df.shape[0]))
#train_df = train_df.drop(labels=too_dark_idx, axis=0)
#train_df = train_df.drop(labels=too_bright_idx, axis=0)
#print('After removing outliers we have {0} training samples.'.format(train_df.shape[0]))

train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

# split, this function returns more than we need as we only need the validation indexes for fastai
tr_n, val_n, tr_idx, val_idx = train_test_split(train_names, range(len(train_names)),
                                                test_size=0.1, stratify=train_labels, random_state=123)


data.shape

tr_n.shape
len(tr_idx)

val_n.shape
len(val_idx)


tr_n
data.iloc[tr_idx,:]

val_n
data.iloc[val_idx,:]

type(val_idx)


# fastai 1.0
from fastai import *
from fastai.vision import *
from torchvision.models import *    # import *=all the models from torchvision  

# specify model architecture, densenet169 seems to perform well for this data but you could experiment
arch = densenet169                  
# specify batch size, hardware restrics this one. Large batch sizes may run out of GPU memory
BATCH_SIZE = 128                    
# input size is the crop size
sz = CROP_SIZE                      
# this will extrat the model name as the model file name e.g. 'resnet50'
MODEL_PATH = str(arch).split()[1]   


# create dataframe for the fastai loader
train_dict = {'name': train_path + train_names, 'label': train_labels}
df = pd.DataFrame(data=train_dict)
# create test dataframe
test_names = []
for f in os.listdir(test_path):
    test_names.append(test_path + f)
df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])

type(df)
df.shape

type(df_test)
df_test.shape


# Subclass ImageList to use our own image opening function
class MyImageItemList(ImageList):
    def open(self, fn:PathOrStr)->Image:
        print(fn)
        print(fn.replace('\\', ''))
        # img = readCroppedImage(fn.replace('/./','').replace('//','/'))
        img = readCroppedImage(fn.replace('\\', ''))
        # This ndarray image has to be converted to tensor before passing on as fastai Image,
        # we can use pil2tensor
        return vision.Image(px=pil2tensor(img, np.float32))


class MyImageItemList(ImageItemList): 
     def open(self, fn:PathOrStr)->Image:
        img = readCroppedImage(fn.replace('./',''))
        return vision.Image(px=pil2tensor(img, np.float32))


# Create ImageDataBunch using fastai data block API
imgDataBunch = (MyImageItemList.from_df(path='/', df=df, suffix='.tif')
        #Where to find the data?
        .split_by_idx(val_idx)
        #How to split in train/valid?
        .label_from_df(cols='label')
        #Where are the labels?
        .add_test(MyImageItemList.from_df(path='/', df=df_test))
        # .add_test(MyImageItemList.from_df(path='', df=df_test))
        #dataframe pointing to the test set?
        .transform(tfms=[[],[]], size=sz)
        # We have our custom transformations implemented in the image loader but we could apply transformations also here
        # Even though we don't apply transformations here, we set two empty lists to tfms. Train and Validation augmentations
        .databunch(bs=BATCH_SIZE)
        # convert to databunch
        .normalize([tensor([0.702447, 0.546243, 0.696453]), tensor([0.238893, 0.282094, 0.216251])])
        # Normalize with training set stats. These are means and std's of each three channel and we calculated these previously in the stats step.
       )

# check that the imgDataBunch is loading our images ok
imgDataBunch.show_batch(rows=2, figsize=(4,4))


# Preso da un commento
# Subclass ImageItemList to use our own image opening function
# =============================================================================
# class MyImageItemList(ImageItemList):
#     def open(self, fn:PathOrStr)->Image:
#         print(fn)
#         tempst=fn.replace('/./','').replace('//','/')
#         img = readCroppedImage(tempst.split('\\')[1])
#         # This ndarray image has to be converted to tensor before passing on as fastai Image,
#         # we can use pil2tensor
#         return vision.Image(px=pil2tensor(img, np.float32))
# =============================================================================

class MyImageItemList(ImageList):
    def open(self, fn:PathOrStr)->Image:
        print(fn)
        tempst=fn.replace('/./','').replace('//','/')
        img = readCroppedImage(tempst.split('\\')[1])
        # This ndarray image has to be converted to tensor before passing on as fastai Image,
        # we can use pil2tensor
        return vision.Image(px=pil2tensor(img, np.float32))



# Idea: tolgo il path da df e df_test
        

# create dataframe for the fastai loader
train_dict = {'name': train_names, 'label': train_labels}
df = pd.DataFrame(data=train_dict)
# create test dataframe
test_names = []
for f in os.listdir(test_path):
#    test_names.append(f)
    test_names.append(f[:-4])
df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])

type(df)
df.shape

type(df_test)
df_test.shape
   

df.shape
df.head() 

df_test.shape
df_test.head()

len(val_idx)
val_idx[:5]
    
class MyImageItemList(ImageList): 
     def open(self, fn:PathOrStr)->Image:
         print(fn)
         img = readCroppedImage(fn.replace('./',''))
         return vision.Image(px=pil2tensor(img, np.float32))  
    
    
imgDataBunch = (
    MyImageItemList.from_df(path=train_path, df=df, suffix='.tif')
    .split_by_idx(val_idx)
    .label_from_df(cols='label')
    .add_test(MyImageItemList.from_df(path=test_path, df=df_test, suffix='.tif'))  
    .transform(tfms=[[],[]], size=sz)
    .databunch(bs=BATCH_SIZE) 
    .normalize([tensor([0.79583625, 0.61863765, 0.78903647]),tensor([0.27073981, 0.31955782, 0.24504378])])
)   
    
    
# =============================================================================
# train_dict = {'name': train_names, 'label': train_labels}
# df = pd.DataFrame(data = train_dict);df.head()
# 
# test_names = []
# for f in os.listdir(test_path):
#     test_names.append(f)
# df_test = pd.DataFrame(np.asarray(test_names), columns=['name']); df_test.head()
# =============================================================================
    


imgDataBunch.show_batch(rows=2, figsize=(4,4))

imgDataBunch.show_batch()
    
    
    
    
    
    
    

