#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from src.data_prep.masks_mker import mask_info, mk_binary_mask, mk_color_mask, mk_textmask
import argparse


# In[2]:


# args

parser = argparse.ArgumentParser(description= 'prep data from test and train dirs to disk for training')
parser.add_argument('-t','--typemask', type = str, metavar = '', required = True, help = 'type of training image to make: all, binary, color, gaug color, gaug text mask')
args = parser.parse_args()


# In[ ]:


# paths
cwd = os.getcwd()
train_csv = os.path.join(cwd, 'data/Train/train.csv') 
test_csv = os.path.join(cwd, 'data/Test/test.csv')
test_img_dir = os.path.join(cwd, 'data/Test')
train_img_dir = os.path.join(cwd, 'data/Train')


# In[ ]:


# load csv / image train data

dfTest = pd.read_csv(test_csv, on_bad_lines = 'skip')
dfTrain = pd.read_csv(train_csv, on_bad_lines = 'skip')


# In[ ]:


# mk all needed files 
def mk_mask(mask_type):
    if mask_type == 'all':
        
        # load info
        test_info = mask_info(test_csv, test_img_dir, transform = None)
        train_info = mask_info(train_csv, train_img_dir, transform = None)

        # mk bi masks for test and train
        mk_binary_mask(test_img_dir, test_info)
        mk_binary_mask(train_img_dir, train_info)

        # mk color masks for test and train
        mk_color_mask(test_img_dir, test_info)
        mk_color_mask(train_img_dir, train_info)

        # mk final masks for test and train
        mk_textmask(test_img_dir, test_info)
        mk_textmask(train_img_dir, train_info)
    
    if mask_type == 'binary':
        
        # load info
        test_info = mask_info(test_csv, test_img_dir, transform = None)
        train_info = mask_info(train_csv, train_img_dir, transform = None)

        # mk bi masks for test and train
        mk_binary_mask(test_img_dir, test_info)
        mk_binary_mask(train_img_dir, train_info)     
        
    if mask_type == 'color':
        
        # load info
        test_info = mask_info(test_csv, test_img_dir, transform = None)
        train_info = mask_info(train_csv, train_img_dir, transform = None)
        
        # mk color masks for test and train
        mk_color_mask(test_img_dir, test_info)
        mk_color_mask(train_img_dir, train_info)
        
    #if mask_type == 'gag_color':
        
    #if mask_type == 'gag_textmask':


# In[ ]:


if __name__ == '__main__':
    mk_mask(args.typemask)

