import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision
from collections import OrderedDict
from torchvision import transforms, models, datasets
import json
from PIL import Image
from torch.autograd import Variable
from torch import nn, optim
import utility
import fmodel

# arch = {"vgg16":25088,
#         "densenet121":1024}
# Create Parse using ArgumentParser
# Creates parse 
parser = argparse.ArgumentParser(description = 'Parser for train.py')

# Creates command line arguments args.dir for path to images files,
# args.arch which CNN model to use for classification, args.labels path to
# text file with names of flowers
parser.add_argument('--data_dir', action="store", default="./flowers/", help = "path to images")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=5, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu")

#get input
args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
architecture = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout

if power == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

def main():
    trainloader, validloader, testloader, train_data = utility.load_data(where)
    model, criterion = fmodel.network_setup(architecture,dropout,hidden_units,lr,power)
    optimizer = optim.Adam(model.classifier.parameters(), lr= 0.001)
    
    # Train Model
    steps = 0
    running_loss = 0
    print_every = 5
    print("**************************")
    print("Training Model in Progress")
    print("**************************")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            if torch.cuda.is_available() and power =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            #forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #define loss steps
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    #fmodel.save_checkpoint(traindata,model,path,struct,hidden_units,dropout,lr)
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :architecture,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    print("Saved checkpoint!")
if __name__== "__main__":
    main()