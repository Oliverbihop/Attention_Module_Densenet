'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os
import argparse

from models import *
from utils import progress_bar

from dataloader import *
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch Custom Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
img_size =[32,32]

# Data
print('==> Preparing data..')
train_data = Image_Loader(root_path="./data_test.csv", image_size=img_size, transforms_data=True)
test_data = Image_Loader_test(root_path="./data_train_s.csv", image_size=img_size, transforms_data=True)
total_train_data = len(train_data)
total_test_data = len(test_data)
print('total_train_data:',total_train_data, 'total_test_data:',total_test_data)

# Generate the batch in each iteration for training and testing
trainloader = DataLoader(train_data, batch_size=8, shuffle=True) # shuffle = true nghĩa là có sáo trộn ảnh khi lấy data bath_size
testloader = DataLoader(test_data, batch_size=8)
# Model
print('==> Building model..')
net = LeNet()
#net = DenseNet121()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

trainning_accuracy=[]
trainning_loss=[]
testing_accuracy=[]
testing_loss=[]
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    trainning_accuracy.append(100.*correct/total)
    trainning_loss.append( train_loss/(batch_idx+1))

        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        testing_accuracy.append(100.*correct/total)
        testing_loss.append(test_loss/(batch_idx+1))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_owndata.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    test(epoch)
    plt.figure(1)
    print(trainning_loss,testing_loss)
    plt.title("Loss-CustomData-dense")
    plt.plot(trainning_loss, color = 'r') # plotting training loss
    plt.plot(testing_loss, color = 'b') # plotting evaluation loss

    plt.legend(['training loss', 'testing loss'], loc='upper left')
    plt.savefig('plot_loss.png')
    #################################################
    plt.figure(2)
    print(trainning_accuracy,testing_accuracy)
    plt.title("Accuracy-CustomData_dense")
    plt.plot(trainning_accuracy, color = 'r') # plotting training loss
    plt.plot(testing_accuracy, color = 'b') # plotting evaluation loss

    plt.legend(['training acc', 'testing acc'], loc='upper left')
    plt.savefig('plot_acc.png')
    plt.show()
    scheduler.step()
