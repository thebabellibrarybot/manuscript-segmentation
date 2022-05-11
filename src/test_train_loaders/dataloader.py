#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL
from PIL import Image, ImageOps
import seg_config
import torch
import glob
import os
import imgaug
import torchvision.transforms as tf


# In[2]:


#df = pd.read_csv(seg_config.Train_GTCSV, on_bad_lines = 'skip')


class dataloaderseg(torch.utils.data.Dataset):
    
    def __init__(self, mask_path, image_path, transform = None):
        self.img_files = glob.glob(os.path.join(image_path, '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(mask_path, os.path.basename(img_path)))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        mask_path = self.img_files[idx]
        img_path = self.mask_files[idx]
        data = (np.array(Image.open(img_path).convert('RGB')))
        mask = (np.array(Image.open(mask_path)))
        
        if self.transform:
            data = self.transform['img_trans'](data)
            mask = self.transform['ann_trans'](mask)
            
        sample = {'img_data': data, 'label_data': mask}
        return sample
        
        
        


# In[3]:


#path = 'Test/img_orgin/'
#for fi in os.listdir(ppath):
#    print(fi)
#    old = os.path.join(ppath, fi)
#    new = old.replace('.jpg','.png')
#    os.rename(old, new)
#    print(old, 'renamed: ', new)


# In[4]:


# check fn before transforms
#ds = dataloaderseg(seg_config.Train_GTIMG, seg_config.Train_GTMASK, transform = None)
#or d in ds:
#    #print(d['img_data'].shape, 'img_data', '\n', d['label_data'].shape, 'label data')
#    label_d = np.unique(d['label_data'])
#    #print(d['label_data'])
#    print(label_d)
    

    
    


# In[5]:


#transformImg = tf.Compose([tf.ToPILImage(),
#                           tf.Resize((224,224)),
#                           tf.ToTensor(),
#                          tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#transformAnn = tf.Compose([tf.ToPILImage(),
#                          tf.Resize((224,224)),
#                          tf.ToTensor()])
#transforms = {'img_trans': transformImg, 'ann_trans': transformAnn}
#print(transforms)
#


# In[6]:


#trans_ds = dataloaderseg(seg_config.Train_GTIMG, seg_config.Train_GTMASK, transform = transforms)
#for d in trans_ds:
#    print(d['img_data'].shape, d['label_data'].shape)


# In[7]:


#trainLoader = DataLoader(trans_ds,
#                         shuffle = True,
#                         batch_size = seg_config.BATCH_SIZE,
#                         pin_memory = seg_config.PIN_MEMORY)
#for t in trainLoader:
#    print(t['img_data'].dtype)
#    print(t['label_data'].dtype)


# In[ ]:




