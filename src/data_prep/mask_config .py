#!/usr/bin/env python
# coding: utf-8

# In[33]:


import os
import pandas as pd


# In[34]:


# file paths
BASE = os.getcwd().split('/')[:-2]
BASE = '/'.join(BASE)
print(BASE)
IMGFOLDER = os.path.join(BASE, 'data/Test/')
COLORMASKFOLDER = os.path.join(BASE, 'data/Test/color_masks/')
BIMASKFOLDER = os.path.join(BASE, 'data/Test/bi_coded_masks/')
CSVFILE = os.path.join(BASE, 'data/Test/test.csv')
IMGOUT = os.path.join(BASE, 'data/Test/imgs/')
FINALMASKOUT = os.path.join(BASE, 'data/Test/masks/')

TRAIN_IMGFOLDER = os.path.join(BASE, 'data/Train/')
TRAIN_COLORMASKFOLDER = os.path.join(BASE, 'data/Train/color_masks/')
TRAIN_BIMASKFOLDER = os.path.join(BASE, 'data/Train/bi_coded_masks/')
TRAIN_CSVFILE = os.path.join(BASE, 'data/Train/train.csv')
TRAIN_IMGOUT = os.path.join(BASE, 'data/Train/imgs/')
TRAIN_FINALMASKOUT = os.path.join(BASE, 'data/Train/masks/')


# In[3]:


# setup 

# current works best for BnW images

THRESHOLD = 225
MAXVALUES = 128


# In[4]:


# label names

LABEL1 = 'margin'
LABEL2 = 'text'
LABEL3 = 'image'
LABEL4 = 'noise'
LABEL5 = 'background'


# In[5]:


# bib ext

# https://www.tutorialspoint.com/how-to-apply-a-mask-on-the-matrix-in-matplotlib-imshow
# https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_masked.html
# https://stackoverflow.com/questions/56766307/plotting-with-numpy-masked-arrays
# https://stackoverflow.com/questions/31877353/overlay-an-image-segmentation-with-numpy-and-matplotlib
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_masked.html
# https://matplotlib.org/3.4.3/tutorials/intermediate/artists.html?highlight=layer%20image

# https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
# https://note.nkmk.me/en/python-pillow-composite/

