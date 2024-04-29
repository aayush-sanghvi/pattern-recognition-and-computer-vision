# Aayush H Sanghvi & Yogi Hetal Shah
# Spring 2024 semseter
# Date:- 4th April 2024
# CS5330- Pattern Recognition and Computer Vision.

import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2


def show_filters(layer):
    for i in range(10):
            # print(model.conv1.weight[i,0])
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        print(layer.weight[i,0])
        plt.imshow(layer.weight[i,0].detach().numpy(), cmap='viridis', interpolation='none')
        plt.title("filter: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_filtered_images(layer,img):
    for i in range(10):
        plt.subplot(5,4,i*2+1)
        plt.imshow(layer.weight[i,0].detach().numpy(), cmap='gray', interpolation='none')
        with torch.no_grad():
            image = cv2.filter2D(img, -1, layer.weight[i,0].detach().numpy())
            plt.subplot(5,4,i*2+2)
            plt.imshow(image, cmap='gray', interpolation='none')
    plt.show()

# main function
# Runs the Extension 1
def main(argv):
    # Load AlexNet model
    model = torchvision.models.alexnet(pretrained=True)
    print(model)
    model.eval()

    # Load training data
    train_dataset = torchvision.datasets.MNIST(root="extention", train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))]))

    # Get first image
    img = cv2.imread('/home/yogi/PRCV/project5/stylus/digits/2.png')
    with torch.no_grad():
        # Get the first convolution layer of the model
        layer1 = model.features[0]
        # print("Filters Layer 1 Weight Shape")
        # print(layer1.weight.shape)
        show_filters(layer1)
        show_filtered_images(layer1,img)

        # Get the second convolution layer of the model
        layer2 = model.features[3]
        # print("Filters Layer 2 Weight Shape")
        # print(layer2.weight.shape)
        show_filters(layer2)
        show_filtered_images(layer2, img)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv)