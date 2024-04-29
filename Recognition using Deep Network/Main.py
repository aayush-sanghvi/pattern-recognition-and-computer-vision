# Aayush H Sanghvi & Yogi Hetal Shah
# Spring 2024 semseter
# Date:- 4th April 2024
# CS5330- Pattern Recognition and Computer Vision.


# import statements
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import numpy
import random
import matplotlib.pyplot as plt

#Number iterations 
n_epochs = 5
# Task 1: Showing first 6 digits -- skimms through the first 6 images of train loader
def task1_a(train_loader):
    examples = enumerate(train_loader)
    batch_idx,(example_data, example_target)=next(examples)
    print(example_data.shape)
    #plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_target[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# class definitions
class MyNetwork(nn.Module):   
    def __init__(self, lr=0.01, ks=5, dr=0.5,log_interval = 10):
        super(MyNetwork, self).__init__()
        self.lr = lr
        self.ks = ks
        self.dr = dr
        self.momentum = 0.5
        self.log_interval = log_interval
        self.conv1 = nn.Conv2d(1, 10, kernel_size=ks)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=ks)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = f.relu(f.max_pool2d(self.conv1(x), 2))
        x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  #Flatterned operation
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        return f.log_softmax(x)
    
    #This function collects the train loader data set and trains the model with this dataset
    #passing throught the output and traget image it predicts the losses and how much data 
    #is proceesed and the accuracy of the current iterations
    def trainings(self,epoch,optimizer,train_loader,train_losses,train_accs,train_counter ,file,printing):   
        self.train()
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = self(data)
            loss = f.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                if printing:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                train_acc = train_correct/len(train_loader.dataset) *100
                train_accs.append(train_acc)
                torch.save(self.state_dict(), '/home/yogi/PRCV/project5/results/'+file+".pth")
                torch.save(optimizer.state_dict(), '/home/yogi/PRCV/project5/results/'+file+"_optimizer.pth")
        print("done training")
        return train_losses,train_counter,train_accs

    #This fuction collects the testing dataset and verifies how close and accuracy is the data to the trained data and prints the accuracy 
    def test(self,test_loader,test_losses,test_accs):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self(data)
                test_loss += f.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_acc = correct / len(test_loader.dataset)
        test_accs.append(test_acc)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * test_acc))
        return test_losses,test_accs

# This is a custom trainer function which takes in the training data and the testing dataset, 
# it trains the model first and simultanously performs test with the trained data 
def network_trainer(model, train_loader, test_loader, n_epochs=3, model_save='model', plot=True):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    train_accs = []
    test_accs = []
    optimizer = optim.SGD(model.parameters(), lr=model.lr, momentum=model.momentum)
    model.test(test_loader, test_losses, test_accs)
    for epoch in range(1, n_epochs + 1):
        train_losses, train_counter, train_accs = model.trainings(epoch,optimizer,train_loader, train_losses,train_accs,train_counter, model_save, True)
        test_losses, test_accs = model.test(test_loader, test_losses, test_accs)

    if plot:
        # print(len(train_losses))
        # print(len(train_accs))
        # print(len(test_losses))
        # print(len(test_acc))
        # print(train_counter)
        # print(test_counter)
        # Plot the training and testing loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_counter, train_losses, label='Training Loss')
        plt.plot(test_counter, test_losses, label='Testing Loss', marker='o', ls='')
        plt.axis()
        plt.title('Training and Testing Loss.')
        plt.figtext(.8, .6,
                    f"L = {model.lr}\nK = {model.ks}\nD = {model.dr}\nB = {train_loader.batch_size}\nE = {n_epochs}")
        plt.xlabel('Number of training examples seen')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot the training and testing accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(train_counter, train_accs, label='Training Accuracy')
        plt.plot(test_counter, test_accs, label='Testing Accuracy', marker='o', ls='')
        plt.title('Training and Testing Accuracy')
        plt.figtext(.8, .6,
                    f"L = {model.lr}\nK = {model.ks}\nD = {model.dr}\nB = {train_loader.batch_size}\nE = {n_epochs}")
        plt.xlabel('Number of training examples seen')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    return train_losses, train_counter, train_accs, test_losses, test_counter, test_accs



# main function (yes, it needs a comment too)
def main(argv):

    # main function code
    print("Starting")
    batch_size_train = 100
    batch_size_test = 100

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/yogi/PRCV/project5/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))])),
                                batch_size=batch_size_train, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/home/yogi/PRCV/project5/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))])),
                                batch_size=batch_size_test, shuffle=False)
  
    #First training
    # start_time = time.time()
    network = MyNetwork()

    # Task 1
    task1_a(train_loader)
    print(network)

    network_trainer(network,train_loader,test_loader,5,'model')
    return


if __name__ == "__main__":
    main(sys.argv)

