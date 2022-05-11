#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torchvision.transforms as tf


# In[2]:


# stuff / out

INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH = 224, 224

BASE_OUT = '/home/mumbot/tombs/fasterRCNN/sem_segment/out/'
MODEL_PATH = os.path.join(BASE_OUT, 'mod.torch')
PLOT_PATH = os.path.join(BASE_OUT, 'plot.png')


# In[3]:


# paths

Test_GTMASK = 'Test/color_masks/'
Test_GTIMG = 'Test/img_orgin/'
Test_GTCSV = 'Test/test.csv'
Train_GTMASK = 'Train/color_masks'
Train_GTIMG = 'Train/img_orgin/'
Train_GTCSV = 'Train/train.csv'
BASE_OUT = 'out/'
Test_PATHS = os.path.join(BASE_OUT, 'tests.txt')


# In[4]:


# device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False 


# In[5]:


# lr, epochs, and batch size

LR = .00001
NUM_EPOCHS = 60
BATCH_SIZE = 3


# In[6]:


# transforms
transformImg = tf.Compose([tf.ToPILImage(),
                           tf.Resize((224,224)),
                           tf.ToTensor(),
                          tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformAnn = tf.Compose([tf.ToPILImage(),
                          tf.Resize((224,224)),
                          tf.ToTensor()])
TRANSFORMS = {'img_trans': transformImg, 'ann_trans': transformAnn}


# In[7]:


# bib


# current use how to seg: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
# use for checker fn: https://imgaug.readthedocs.io/en/latest/source/installation.html#installation-in-pip
# info on thresholds: https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/
# weight thresholding: https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/?utm_source=blog&utm_medium=image-segmentation-article
# ^^^^ *** ^^^^
# future use how to ocr: https://deepayan137.github.io/blog/markdown/2020/08/29/building-ocr.html


# In[ ]:




