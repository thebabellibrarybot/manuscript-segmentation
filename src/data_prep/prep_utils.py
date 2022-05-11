#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os, fnmatch
from sklearn.model_selection import train_test_split
import shutil


# In[16]:


IMGPATH = '/home/mumbot/tombs/bboxtomb/bbox_imgs/'
csv_file = '/home/mumbot/tombs/bboxtomb/bbox_imgs/full_CSV.csv'


# In[20]:


# mk test / train split from full image file and single csv annotation

# mk dirs 

testdir = 'Test'
traindir = 'Train'
trainlist = []
testlist = []

def prep_test_train_dirs(orgin_img_path):
    orgin_img_path = orgin_img_path
    try:
        os.mkdir(testdir)
        print("Directory ", testdir, " Created")
    except FileExistsError:
        print("Directory ", testdir, " already exsists")
    try: 
        os.mkdir(traindir)
        print("Directory ", traindir, " Created")
    except FileExistsError:
        print("Directory ", traindir, " already exsists")

    # split ls of img_files

    try: 
        imglist = fnmatch.filter(os.listdir(orgin_img_path), '*.jpg')
        split = train_test_split(imglist, test_size = 0.20, random_state = 42)
        (trainlist, testlist) = split
        #trainlist.append(trainlist)
        #testlist.append(testlist)
    except ValueError:
        print("already split")

    # add images to new folder spaces
    try: 
        for ls in trainlist:
            trainpath = '/home/mumbot/tombs/bboxtomb/Train/'
            oldpath = os.path.join(IMGPATH, ls)
            shutil.move(oldpath, trainpath)
            print((oldpath, " moved to ", trainpath))
        for ls in testlist:
            testpath = '/home/mumbot/tombs/bboxtomb/Test/'
            oldpath = os.path.join(IMGPATH, ls)
            shutil.move(oldpath, testpath)
    except ValueError:
        print(oldpath, " already moved")

        return print("Paths to test train created and image files moved.")


# In[21]:


prep_test_train_dirs(IMGPATH)


# In[43]:


# split csv file and add to each new folder (will cont to add each call)

train_path = '/home/mumbot/tombs/bboxtomb/Train'
test_path = '/home/mumbot/tombs/bboxtomb/Test'
train_csv = '/home/mumbot/tombs/bboxtomb/Train/train.csv'
test_csv = '/home/mumbot/tombs/bboxtomb/Test/test.csv'
train_ls = []
test_ls = []
train_row = []
test_row = []

def split_orgin_csv(trainpath, testpath):
    
    trainpath = train_path
    testpath = test_path
    
    train_img_ls = fnmatch.filter(os.listdir(train_path), '*.jpg')
    for ls in train_img_ls:
        train_img_ls = ls.split(".jpg")
        train_ls.append(train_img_ls[0])
    test_img_ls = fnmatch.filter(os.listdir(test_path), '*.jpg')
    
    for ls in test_img_ls:
        test_img_ls = ls.split(".jpg")
        test_ls.append(test_img_ls[0])
    train_tup = tuple(train_ls)
    test_tup = tuple(test_ls)


    # write rows of annotations that aligns with image data into each new csv file
    # probably need to add an arg to stop from rewriting lines

    rows = open(csv_file).read().strip().split("\n")
    for row in rows:
        if row.startswith(train_tup):
            with open(train_csv, 'a') as f_obj:
                f_obj.write(row)
                f_obj.write('\n')
                f_obj.close
    for row in rows:
        if row.startswith(test_tup):
            with open(test_csv, 'a') as fi_obj:
                fi_obj.write(row)
                fi_obj.write('\n')
                fi_obj.close


# In[44]:


split_orgin_csv(train_path, test_path)


# In[ ]:




