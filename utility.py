#import all the libraries necessary
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from collections import OrderedDict
from torchvision import transforms, models, datasets
import json
from PIL import Image
from torch.autograd import Variable

'''
this project helps to load the data set required to be trained 
transforms the training, testing and validation sets using transforms
Load the transforms sets intor
'''

#define the method for loading dataset
def load_data( data_dir= "./flowers"):
    #read the data names to a json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    #create a variable to navigate to each directory of file
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    ## transform training data set
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    # transform validation data set
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    # transform testing  data set
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle = True)

    return trainloader, validloader, testloader, train_data

# loader = load_data(data_dir = testloader)
# # change this to the trainloader or testloader 
# data_iter = iter(testloader)

# images, labels = next(data_iter)
# fig, axes = plt.subplots(figsize=(10,4), ncols=4)
# for ii in range(4):
#     ax = axes[ii]
#     helper.imshow(images[ii], ax=ax, normalize=True)