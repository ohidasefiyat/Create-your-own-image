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

# Create Parse using ArgumentParser
# Creates parse 
parser = argparse.ArgumentParser(description = 'Parser for predict.py')

parser.add_argument('input', default='./flowers/test/10/image_07090.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

#define all the variables needed to for prediction
args = parser.parse_args()
path_image = args.input
output_count = args.top_k
device = args.gpu
json_name = args.category_names
path = args.checkpoint

#define function to help with prediction
def main():
    model=fmodel.load_checkpoint(path)
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)
        
    probs = fmodel.predict(path_image, model, output_count, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
    prob = np.array(probs[0][0])
    i=0
    while i < output_count:
        print("{} with a probability of {}".format(labels[i], prob[i]))
        i += 1
    #fmodel.plot_solution(path_image, list(fmodel.arch.values())[0] )
    

    
if __name__== "__main__":
    main()