# Aayush H Sanghvi & Yogi Hetal Shah
# Spring 2024 semseter
# Date:- 4th April 2024
# CS5330- Pattern Recognition and Computer Vision.


from Main import network_trainer, MyNetwork
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import numpy
import random
import cv2
import matplotlib.pyplot as plt


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )


def main(argv):
    # handle any command line arguments in argv

    # main function code
    print("Starting")
    n_epochs = 10
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 5

    #utilizing the perviously trained data 
    model = MyNetwork(log_interval=log_interval)
    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
    model_stat_dict = torch.load('/home/yogi/PRCV/project5/results/model.pth')
    model.load_state_dict(model_stat_dict)
    model_optimizer_dict = torch.load('/home/yogi/PRCV/project5/results/model_optimizer.pth')
    model_optimizer.load_state_dict(model_optimizer_dict)

    # freezes the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False
    
    #replacing the last layer
    model.fc2 = nn.Linear(50,5)
    print(model)

    for param in model.parameters():
        param.requires_grad = True
    
    # DataLoader for the Greek data set
    training_set_path="/home/yogi/PRCV/project5/greek_custom"
    greek_train_custom = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( training_set_path,
                    transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                    GreekTransform(),torchvision.transforms.Normalize((0.1307,), (0.3081,) ) ] ) ),
                    batch_size = 5,shuffle = True )
    
    greek_test = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( training_set_path,
                    transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                    GreekTransform(),torchvision.transforms.Normalize((0.1307,), (0.3081,) ) ] ) ),
                    batch_size = 5,shuffle = True )
    
    greek_test_custom = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( training_set_path,
                    transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                    GreekTransform(),torchvision.transforms.Normalize((0.1307,), (0.3081,) ) ] ) ),
                    batch_size = 5,shuffle = True )
    
    #Training the Data
    network_trainer(model,greek_train_custom,greek_test_custom,n_epochs,"greek_custom",plot=True)
    examples = enumerate(greek_test)
    
    fig1 = plt.figure()
    for i in range(5):
        batch_idx,(example_data,example_targets) = next(examples)
        with torch.no_grad():
            output = model(example_data)

        plt.subplot(2,3,i+1)
        plt.tight_layout()
        if (output.data.max(1, keepdim=True)[1][i].item() == 0):
            val = 'alpha'
        elif(output.data.max(1, keepdim=True)[1][i].item() == 1):
            val = 'beta'
        elif(output.data.max(1, keepdim=True)[1][i].item() == 2):
            val = 'gamma'
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(val))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    return

if __name__ == "__main__":
    main(sys.argv)

