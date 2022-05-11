#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tf 
import seg_config
import dataloader
from dataloader import dataloaderseg
import pynvml
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


# load data from dataloader notebook as test and train:

# TODO: ADD MORE TF FOR THINGS LIKE RANDOMCONTRAST,CROP,ECT.
transformImg = tf.Compose([tf.ToPILImage(),
                           tf.Resize((224,224)),
                           tf.RandomEqualize(.15),
                           tf.RandomAutocontrast(.15),
                           tf.RandomAdjustSharpness(1.25, .15),
                           tf.RandomAdjustSharpness(.75, .15),
                           tf.ToTensor(),
                          tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # i think i need to custom this
transformAnn = tf.Compose([tf.ToPILImage(),
                          tf.Resize((224,224)),
                           tf.RandomEqualize(.15),
                           tf.RandomAutocontrast(.15),
                           tf.RandomAdjustSharpness(1.25, .15),
                           tf.RandomAdjustSharpness(.75, .15),
                           tf.ToTensor()])
transforms = {'img_trans': transformImg, 'ann_trans': transformAnn}

# mk test,train loaders:
train_ds = dataloaderseg(seg_config.Train_GTIMG, seg_config.Train_GTMASK, transform = transforms)
test_ds = dataloaderseg(seg_config.Test_GTIMG, seg_config.Test_GTMASK, transform = transforms)
trainLoader = DataLoader(train_ds,
                         shuffle = True,
                         batch_size = seg_config.BATCH_SIZE
                        )
testLoader = DataLoader(test_ds,
                         shuffle = True,
                         batch_size = seg_config.BATCH_SIZE
                        )

for i, d in enumerate(trainLoader):
    print(d['label_data'].shape)
    print(d['img_data'].shape)

    


# In[3]:


# loading nn  *deeplabv3_resnet50*:

Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = True)
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size = (1, 1), stride = (1, 1))
Net = Net.to(seg_config.DEVICE)
Optimizer = torch.optim.Adam(params = Net.parameters(), lr = seg_config.LR)
criterion = torch.nn.CrossEntropyLoss()


# In[4]:


# setup print statements for model eval
print('[INFO] training the network... ')
startTime = time.time()
trainSteps = len(train_ds) // seg_config.BATCH_SIZE
testSteps = len(test_ds) // seg_config.BATCH_SIZE
H = {'train_loss': [], 'test_loss': []}

# train loop
for e in tqdm(range(seg_config.NUM_EPOCHS)):
    print(e)
    Net.train()
    totaltrainloss = 0
    totaltestloss = 0
    for i, d in enumerate(trainLoader):
        # load to model for pred
        img = d['img_data']
        label = d['label_data']
        label = torch.argmax(label, dim=1) # adjust shape for CEL

        #img, label = img.to(seg_config.DEVICE), label.to(seg_config.DEVICE)
        img = torch.autograd.Variable(img, requires_grad = False).to(seg_config.DEVICE)
        label = torch.autograd.Variable(label,requires_grad = False).to(seg_config.DEVICE)
        Pred = Net(img)['out']

        Loss = criterion(Pred, label.long())
        Loss.backward()
        Optimizer.step()

        totaltrainloss += Loss
        
    # eval loop
    with torch.no_grad():
        Net.eval()
        for i, d in enumerate(testLoader):
            # load to model for pred
            img, label = d['img_data'], d['label_data']
            label = torch.argmax(label, dim=1) # adjust shape for cel
            img = torch.autograd.Variable(img, requires_grad = False).to(seg_config.DEVICE)
            label = torch.autograd.Variable(label,requires_grad = False).to(seg_config.DEVICE)
            Pred = Net(img)['out']
            
            Loss = criterion(Pred, label)
            totaltestloss += Loss
            
    avgtrainLoss = totaltrainloss / trainSteps
    avgtestLoss = totaltestloss / testSteps
    H['train_loss'].append(avgtrainLoss.cpu().detach().numpy())
    H['test_loss'].append(avgtestLoss.cpu().detach().numpy())
    print('[INFO] Epoch: {}/{}'.format(e + 1, seg_config.NUM_EPOCHS))
    print('Train loss: {:.6f}, Test loss {:.4f}'.format(avgtrainLoss, avgtestLoss))
    if e == 40:
        break
        
endTime = time.time()
print('[INFO] total time taken to train the model: {:.2f}s'.format(endTime - startTime))


# In[6]:


# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(seg_config.PLOT_PATH)
# serialize the model to disk
torch.save(Net, seg_config.MODEL_PATH)
print('model saved to: ', seg_config.MODEL_PATH)


# In[ ]:


# notes on CEL: https://discuss.pytorch.org/t/runtimeerror-only-batches-of-spatial-targets-supported-3d-tensors-but-got-targets-of-dimension-4/82098/8

# tips on training: https://discuss.pytorch.org/t/runtimeerror-only-batches-of-spatial-targets-supported-3d-tensors-but-got-targets-of-dimension-4/82098/8

# building deeplabv3 nn: https://towardsdatascience.com/train-neural-net-for-semantic-segmentation-with-pytorch-in-50-lines-of-code-830c71a6544f
# building training print methods: https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/


# In[ ]:




