import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from dataloader import *

'''
[aVgNet] Audio Visibility Graph Network
'''

class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, apply_pool=True):
        super(ComplexConv,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding
        self.apply_pool = apply_pool
        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.apply_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x): # x : [batch,2,nch,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        real = F.relu(real)
        if self.apply_pool:
            real = self.pool(real)
        imag = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        imag = F.relu(imag)
        if self.apply_pool:
            imag = self.pool(imag)
        output = torch.stack((real,imag),dim=1)
        return output

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## Model components
        self.linear_re = nn.Linear(in_features, out_features)
        self.linear_im = nn.Linear(in_features, out_features)

    def forward(self, x): # x : [batch,2,nch,axis1,axis2]
        real = self.linear_re(x[:,0]) - self.linear_im(x[:,1])
        imag = self.linear_re(x[:,1]) + self.linear_im(x[:,0])
        output = torch.stack((real,imag),dim=1)
        return output


class aVgNet(nn.Module):
    def __init__(self):
        super(aVgNet, self).__init__()
        
        self.conv1 = ComplexConv(in_channel=4, out_channel=6, kernel_size=(5, 5), apply_pool=False)
        self.conv2 = ComplexConv(in_channel=6, out_channel=9, kernel_size=(5, 5))
        
        self.fc = ComplexLinear(in_features=9*252*9, out_features=9*32*32)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 2, 9*252*9)
        x = self.fc(x)
        x = x.view(-1, 2, 9, 32, 32)
        return x

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
# Parameters
params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}

max_epochs = 100
dataset_file = "/scratch/data/LOCATA/MIC_eval_eigenmike_16k/train.hdf"

# Generators
training_set = HDF5AudioDataset(dataset_file)
training_generator = torch.utils.data.DataLoader(training_set, **params)

# validation_set = 
# validation_generator = 

# aVgNet model initialization 
avgnet_model = aVgNet()
avgnet_model = avgnet_model.to(device)
# Training parameters
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(avgnet_model.parameters(), lr=0.00001)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    batch_count = 0
    for data_batch, labels in training_generator:
        # Transfer to GPU
        data_batch, labels = torch.as_tensor(data_batch, dtype=torch.float32).to(device), torch.as_tensor(labels, dtype=torch.float32).to(device)
        # Forward pass
        outputs = avgnet_model(data_batch)
        loss = criterion(outputs, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Stats
        running_loss += loss.item()
        if batch_count % 10 == 9:    # Print every 10 mini-batches
            print('[%d, %5d] loss: %.8f' % (epoch + 1, batch_count + 1, running_loss / 100))
            running_loss = 0.0
        batch_count += 1

print('Finished Training aVgNet')

# To be done (TBD) up next:
'''
# Validation
with torch.set_grad_enabled(False):
    for data_batch, local_labels in validation_generator:
        # Transfer to GPU
        data_batch, local_labels = data_batch.to(device), local_labels.to(device)

        # Model computations
        [...]
'''