'''LeNet in PyTorch.'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
from models.cbam import CBAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
using_cbam=True
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.cbam = CBAM(6,2)
        self.cbam1 = CBAM(16,2)
    def forward(self, x):
        print(x.shape)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        if using_cbam:
          out = self.cbam(out)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        if using_cbam:
          out=  self.cbam1(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
import time
def test():
    net = LeNet()
    x = torch.randn(1,3,32,32)
    t = time.time()
    y = net(x)
    print(time.time()-t)
    #summary(net, (3, 32, 32))
    
test()
