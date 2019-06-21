# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:57:00 2019

@author: leona
"""


# Prova 1
# pickle

import pickle

x = 2

with open("prova_pickle_1", "wb") as file:
  pickle.dump(x, file)

with open("prova_pickle_1", "rb") as file:
  y = pickle.load(file)


y



# Read baseline model
# fastai 1.0
from fastai import *
from fastai.vision import *
from torchvision.models import *    # import *=all the models from torchvision  

arch = densenet169                  # specify model architecture, densenet169 seems to perform well for this data but you could experiment
BATCH_SIZE = 128                    # specify batch size, hardware restrics this one. Large batch sizes may run out of GPU memory
sz = CROP_SIZE                      # input size is the crop size
MODEL_PATH = str(arch).split()[1]   # this will extrat the model name as the model file name e.g. 'resnet50'

# Next, we create a convnet learner object
# ps = dropout percentage (0-1) in the final layer
def getLearner():
    return create_cnn(imgDataBunch, arch, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

# Scarica da internet i pesi del learner
learner = getLearner()



learner.load(MODEL_PATH + '_stage1')


