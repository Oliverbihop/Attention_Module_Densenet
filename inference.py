# from lib import *
# from densenet_re import *
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models import *
import matplotlib.pyplot as plt
from support import *
import numpy as np
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
import os
import torchvision.transforms as transforms
import torchvision
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])





# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


    # Load checkpoint.

assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

test_data = Image_Loader_test(root_path="./data_train_s.csv", image_size=img_size, transforms_data=True)

testloader = DataLoader(test_data, batch_size=1)
                        
total_test_data = len(test_data)
print( 'total_train_data:',total_test_data)

#TRAINING
k_true_img0 = []
k_true_img1 = []
k_pred_out0 = []
k_pred_out1 = []

k_true = []
k_pred = []
print('=======> Start Training:')
iters = 0
for index, data in enumerate(test_loader):
    iters = iters + 1
    image, label = data
    for index, data in enumerate(label):
      k_true.append(int(data.numpy()))
    image = image.to(device)
    label = label.to(device)
    y_pred = net(image)

    _, pred = y_pred.max(1)
    print(pred)
    k_pred.append(int(pred))
    
    # print("==========")
    # print("k_true: ", k_true)
    # print("k_pred: ",k_pred) 
    # print("==========")
print("k_true: ", len(k_true))
print("k_pred: ",len(k_pred) )
#print('Accuracy = ',accuracy_score(k_true, k_pred))

cnf = confusion_matrix(k_true,k_pred) 
#print('Accuracy = ',np.diagonal(cnf).sum()/cnf.sum())
class_names = ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6' ]
plt.figure()
plot_confusion_matrix(cnf, classes=class_names,title='Confusion matrix')
print('=================================')
