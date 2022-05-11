#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this will be used to make masks from BnW text baised images processed as binary masks


# In[2]:



import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
import argparse
import sys


#train_csv = '/home/mumbot/tombs/fasterRCNN/deeplabv3_L_DS/data/Train/train.csv' 
#test_csv = '/home/mumbot/tombs/fasterRCNN/deeplabv3_L_DS/data/Test/test.csv'
#test_img_dir = '/home/mumbot/tombs/fasterRCNN/deeplabv3_L_DS/data/Test'
#train_img_dir = '/home/mumbot/tombs/fasterRCNN/deeplabv3_L_DS/data/Train'
#dfTest = pd.read_csv(test_csv, on_bad_lines = 'skip')
#dfTrain = pd.read_csv(train_csv, on_bad_lines = 'skip')


# In[3]:



class mask_info(torch.utils.data.Dataset):
    
    def __init__(self, csv, root, transform):
        self.root = root
        self.csv = pd.read_csv(csv, on_bad_lines = 'skip')
        self.transform = transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # list of info to find
        imgls = []
        boxesls = []
        labelsls = []
        
        # calling info
        img = os.path.join(self.root, self.csv.iloc[idx, 0])
        boxes = self.csv.iloc[idx, 5].split(',')
        x = boxes[1].split(':')[1]
        y = boxes[2].split(':')[1]
        w = boxes[3].split(':')[1]
        h = boxes[4].split(':')[1].split('}')[0]
        boxes = x, y, w, h
        labels = self.csv.iloc[idx, 6].split(':')[1].split('"')[1]
        num_labels = self.csv.iloc[idx, 3]
        
        # append info to list so there is only one list per filename
        labelsls.append(labels)
        boxesls.append(boxes)        
        imgls.append(img)
        for i in range(num_labels):
            if i >= 1:
                img = os.path.join(self.root, self.csv.iloc[idx + i, 0])
                imgls.append(img)
                boxes = self.csv.iloc[idx + i, 5].split(',')
                x = boxes[1].split(':')[1]
                y = boxes[2].split(':')[1]
                w = boxes[3].split(':')[1]
                h = boxes[4].split(':')[1].split('}')[0]
                boxes = x, y, w, h
                boxesls.append(boxes)        
                labels = self.csv.iloc[idx + i, 6].split(':')[1].split('"')[1]
                labelsls.append(labels)
                
# return important info aka: imgpath, box dims, labels, and number of labels per file
       
        targets = {}
        targets['img_name'] = imgls
        targets['boxes'] = boxesls
        targets['labels'] = labelsls
        targets['num_labels']= num_labels
        sample = img, targets
        return sample 


# In[4]:


def mk_binary_mask(img_orgin_path, dataset):
    
    # find and make directory for binary image masks
    
    mask_path = os.path.join(img_orgin_path, 'bimasks')
    os.mkdir(mask_path)
    print('mask path = ' + mask_path)
    
    # find data to parse for bindary image masks
    
    for i, d in dataset:
        realname = d['img_name'][0]
        num_labels = len(d['labels'])
        if (d['img_name'][num_labels-1]) == realname:
            print('filtering data for: ' + realname)
            
            # mk new title from old name, apply filter, save to mask path directory
            
            maxval = 225
            thresh = 128
            img_data = Image.open(realname).convert('L')
            img_invert = ImageOps.invert(img_data)
            img_gray = np.array(img_invert)
            img_bin = (img_gray > thresh) * maxval
            savetitle = realname.split('/')
            savetitle = savetitle[8]
            savetitle = savetitle.replace('.jpg', '.png')
            savetitle = 'mask_' + savetitle
            masktitle = os.path.join(mask_path, savetitle)
            
            Image.fromarray(np.uint8(img_bin)).save(masktitle, 'PNG')
            print('saved binary image file: ' + savetitle + '\n' + ' to file location: ' + mask_path)
            


# In[5]:


#test_info = mask_info(test_csv, test_img_dir, transform = None)
#train_info = mask_info(train_csv, train_img_dir, transform = None)

#mk_binary_mask(test_img_dir, test_info)
#mk_binary_mask(train_img_dir, train_info)


# In[6]:


def mk_color_mask(img_orgin_path, dataset):
    
    # find and make directory for binary image masks
    
    colormask_path = os.path.join(img_orgin_path, 'color_masks')
    os.mkdir(colormask_path)
    print('color mask path = ' + colormask_path)
    
    # find data to parse for bindary image masks
    
    for i, d in dataset:
        realname = d['img_name'][0]
        num_labels = len(d['labels'])
        if (d['img_name'][num_labels-1]) == realname:
            print('filtering data for: ' + realname)
            
            # mk new title from old name, apply filter, save to mask path directory
            
            # open base image for data
            img_data = Image.open(realname)
            H, W = img_data.height, img_data.width
            img_data.close()
            
            # open base drawing pad
            image = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(image) 
            
            # save path
            svfi_name = 'colormask_' + realname.split('/')[8]
            sv_name = os.path.join(os.path.join(colormask_path, svfi_name))
            sv_name = sv_name.replace('.jpg', '.png')
            print('saving data for '+ svfi_name + ' to: ' + sv_name)
                            
            # draw label bbox as indvidual shape colors
            
            # margin
            for i in range(num_labels):
                img, box, label = (d['img_name'][0], d['boxes'][i], d['labels'][i])
                x,y,h,w = box
                x,y,h,w = float(x), float(y), float(h), float(w)   
                if label == 'margin':
                    color = 1
                    x2 = x + h
                    y2 = y + w
                    draw.rectangle((x, y, x2, y2), fill=color)
                    print('drawing ' + label + ' for ' + svfi_name)
            
            # text
            for i in range(num_labels):
                img, box, label = (d['img_name'][0], d['boxes'][i], d['labels'][i])
                x,y,h,w = box
                x,y,h,w = float(x), float(y), float(h), float(w)  
                if label == 'text':
                    color1 = 2
                    x2 = x + h
                    y2 = y + w
                    draw.rectangle((x, y, x2, y2), fill = color1)
                    print('drawing ' + label + ' for ' + svfi_name)
            
            # image
            for i in range(num_labels):
                img, box, label = (d['img_name'][0], d['boxes'][i], d['labels'][i])
                x,y,h,w = box
                x,y,h,w = float(x), float(y), float(h), float(w)  
                if label == 'image':
                    color = 3
                    x2 = x + h
                    y2 = y + w
                    draw.rectangle((x, y, x2, y2), fill = color)
                    print('drawing ' + label + ' for ' + svfi_name)
                    
            image.save(sv_name, 'PNG')
            print(sv_name, ': saved to ', colormask_path)
            print('finished item: ' + str(i))
      


# In[ ]:





# In[7]:


def mk_textmask(img_orgin_path, dataset):
    
    # find and make directory for binary image masks
    
    mask_path = os.path.join(img_orgin_path, 'masks')
    os.mkdir(mask_path)
    print('mask path = ' + mask_path)
    
    # find data to parse for bindary image masks

    color_img_dir = os.listdir(os.path.join(img_orgin_path, 'colormasks/'))
    bi_img_dir = os.listdir(os.path.join(img_orgin_path, 'bimasks/'))
    for i in range(len(color_img_dir)):
        i1 = color_img_dir[i]
        i2 = i1.replace('colormask_', 'mask_')
        im1 = Image.open(os.path.join(os.path.join(img_orgin_path, 'colormasks'), i1))
        im1array = np.array(im1)
        H, W = im1.heigh, im1.width
        im2 = Image.new('L', (W, H), 0)
        im2array = np.array(im2)
        im2.save('im2', 'PNG')
        im2 = Image.open('im2')
        im2arrayb = np.array(im2)
        mask = Image.open(os.path.join(os.path.join(img_orgin_path, 'bimasks'), i2))
        final_im = Image.composite(im1, im2, mask)
        fi_name = color_img_dir[i].split('_')
        fi_name = fi_name[1]
        final_sv = os.path.join(mask_path, fi_name)
        final_im.save(final_sv, 'PNG')
        


# In[ ]:





# In[8]:


# bio: 

# gag blur: https://www.tutorialspoint.com/python_pillow/python_pillow_blur_an_image.htm
# gag blur: https://www.google.com/search?q=gauggasian+blue+in+pil+so+the+image+gets+brighter+&ei=c2JjYq28OuPMptQPzaWBuAk&ved=0ahUKEwjtwZfoj6n3AhVjpokEHc1SAJcQ4dUDCA8&uact=5&oq=gauggasian+blue+in+pil+so+the+image+gets+brighter+&gs_lcp=Cgdnd3Mtd2l6EAMyBwghEAoQoAEyBwghEAoQoAEyBwghEAoQoAEyBQghEKsCMgUIIRCrAjIFCCEQqwI6BwgAEEcQsANKBAhBGABKBAhGGABQ_w5YyBRgzyJoAXABeACAAWKIAfAFkgEBOZgBAKABAcgBCMABAQ&sclient=gws-wiz
#


# In[9]:


# prac blurring all color masks

#def blur(folder_path):
#    fp = self.folder_path
#    for fi in os.listdir(fp):
#        print(fi)
#blur()


# In[10]:


#imgs = os.listdir(mask_config.FINALMASKOUT)
#fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (30, 14), dpi = 100, sharex = True, sharey = True)



#for i, img in enumerate(imgs):
#    imgn = os.path.join(mask_config.FINALMASKOUT, img)
#    img = Image.open(imgn)
#    imgB = img.filter(ImageFilter.GaussianBlur(15))
    
    #fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (15, 7), dpi = 80, sharex = True, sharey = True)
 #  if i == 0:
 #      ax[0, 0].imshow(img)
 #      ax[0, 1].imshow(imgB)
 #   if i == 1:
 #       ax[1, 0].imshow(img)
 #       ax[1, 1].imshow(imgB)
 #   if i == 2:
 #       ax[2, 0].imshow(img)
 #       ax[2,1].imshow(imgB)
        
 


# In[ ]:




