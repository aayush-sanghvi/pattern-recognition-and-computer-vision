# Aayush H Sanghvi & Yogi Hetal Shah
# Spring 2024 semseter
# Date:- 4th April 2024
# CS5330- Pattern Recognition and Computer Vision.


import Main
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt

#this is a part of the task where it shows the first 6 images of the testing dataset
def task1(test_loader):
    examples = enumerate(test_loader)
    batch_idx,(example_data, example_target)=next(examples)
    print(example_data.shape)
    plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_target[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# This is task1 - e which predicts the testing images based on the model and gives the classification of the number
def task1_e(test_loader,model):
    examples = enumerate(test_loader)
    for i in range(9):
        batch_idx,(example_data,example_targets) = next(examples)
        with torch.no_grad():
            output = model(example_data)
            
        rounded_output = torch.round(output[i] * 100)/100
        print(rounded_output)
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        print
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    # for i in range(len(output)):
    #     print(torch.round(output[i] * 100)/100)
    plt.show()


#This is task1_f where in it imports new custom loader dataset pf hand written Numbers 
def task1_f(custom_loader,model):
    examples = enumerate(custom_loader)
    batch_idx,(example_data, example_target)=next(examples)
    print(example_data.shape)
    # custom_testlosses = Main.test(customtest_loader,continued_network)
    test_loss = 0
    i = 0
    fig2 = plt.figure()
    with torch.no_grad():
        for data, target in custom_loader:
            output = model(data)
            test_loss += f.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]

    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(pred[i][0].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main(argv):
    # handle any command line arguments in argv

    # main function code
    print("Starting")
    batch_size_test = 1000


    #utilizing the perviously trained data 
    continued_network = Main.MyNetwork()
    
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/yogi/PRCV/project5/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))])),
                                batch_size=batch_size_test, shuffle=False)

    network_state_dict = torch.load('/home/yogi/PRCV/project5/results/model.pth')
    continued_network.load_state_dict(network_state_dict)
    
    #Task1
    #task1(test_loader)

    #Task1- F
    task1_e(test_loader,continued_network)

    directory = "/home/yogi/PRCV/project5/photos"
    customtest_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(directory,transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((28,28)), torchvision.transforms.Grayscale(),
        torchvision.transforms.functional.invert,torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size = 10,shuffle = False)

        #Task1- g
    task1_f(customtest_loader,continued_network)
    
    return


if __name__ == "__main__":
    main(sys.argv)

