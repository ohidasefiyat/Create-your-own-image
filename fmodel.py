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
from torch import nn, optim
import utility

'''
this project helps to build the network model using vgg16 and densenet161
create checkpoint to save the model
load the checkpoint in .pth file
process the images in order to scale, crops and normalize the image
predict the image giving it appropriate probability
'''

arch = {"vgg16":25088,
        "densenet121":1024}

#define the network architectur 
def network_setup(architecture='vgg16',dropout=0.1,hidden_units=4096, lr=0.001, device='gpu'):
    #use cuda if available else use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #select the architecture to use
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    #Freeze parameters so we don't backprop through them
    for para in model.parameters():
        para.requires_grad = False
      
    #replace the classifier since the layer was trained on imageNet
    model.classifier = nn.Sequential(nn.Linear(arch['vgg16'], hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    #Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    print(model)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    
    return model, criterion



'''
define the method to save the model to .pth file this avoid retraining of 
the model and make it easily accessible to consume
'''
#
def save_checkpoint(train_data, model = 0, path = 'checkpoint.pth', architecture = 'vgg16', hidden_units = 4096, dropout = 0.2, lr = 0.001, epochs = 1):
    #map classes to indices gotten from one of the image datasets
    model.class_to_idx =  train_data.class_to_idx
    #save the checpoint as .pth
    torch.save({'structure' :architecture,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)

    
    
'''
 function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.
'''  
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_path ='checkpoint.pth' ):
    #define the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    lr=checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    architecture = checkpoint['structure']
    
    model,_ = network_setup(architecture, dropout,hidden_units,lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    image = image_transforms(pil_image)
    
    return image



def predict(image_path, model, topk=5, device = 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    model.to('cuda')
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        logps = model.forward(img.cuda())
        
    probability = torch.exp(logps).data
    
    return probability.topk(topk)

# Display an image along with the top 5 classes
def plot_solution(path, model):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    index = 1
    #path = test_dir + '/1/image_06764.jpg'
    ps = predict(path, model)
    image = process_image(path)

    ax1 = imshow(image, ax = plt)
    ax1.axis('off')
    ax1.title(cat_to_name[str(index)])


    a = np.array(ps[0][0])
    b = [cat_to_name[str(index+1)] for index in np.array(ps[1][0])]

    fig,ax2 = plt.subplots(figsize=(5,5))


    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(b)
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()
    ax2.barh(y_pos, a)

    return plt.show()