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
import numpy
import random
import cv2
import matplotlib.pyplot as plt


def main(argv):
    # handle any command line arguments in argv

    # main function code
    print("Starting")
    batch_size_train = 48
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    #utilizing the perviously trained data 
    model = Main.MyNetwork()
    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
    model_stat_dict = torch.load('/home/yogi/PRCV/project5/results/model.pth')
    model.load_state_dict(model_stat_dict)
    model_optimizer_dict = torch.load('/home/yogi/PRCV/project5/results/model_optimizer.pth')
    model_optimizer.load_state_dict(model_optimizer_dict)
    

    print(model)
    
    print("----------Part A-------------")
    for i in range(10):

        # print(model.conv1.weight[i,0])
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        print(model.conv1.weight[i,0])
        plt.imshow(model.conv1.weight[i,0].detach().numpy(), cmap='viridis', interpolation='none')
        plt.title("filter: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    print("----------Part B-------------")
    img = cv2.imread('/home/yogi/PRCV/project5/stylus/digits/2.png')
    for i in range(10):
        plt.subplot(5,4,i*2+1)
        plt.imshow(model.conv1.weight[i,0].detach().numpy(), cmap='gray', interpolation='none')
        with torch.no_grad():
            image = cv2.filter2D(img, -1, model.conv1.weight[i,0].detach().numpy())
            plt.subplot(5,4,i*2+2)
            plt.imshow(image, cmap='gray', interpolation='none')
    plt.show()




    return

if __name__ == "__main__":
    main(sys.argv)

