import torch
import torch.nn as nn
import math
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5,2,padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(16,eps=1e-5)
        self.dropout = nn.Dropout(p=0.2)
        self.block1 = nn.Sequential(
            nn.Conv2d(16,32,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.pool = nn.MaxPool2d(2,2)
        layers = [[nn.Conv2d(32,32,1,1),nn.LeakyReLU(0.1)] for _ in range(5)]
        self.layers=[]
        for layer in layers:
            self.layers+=layer
        self.block2 = nn.Sequential(*self.layers)
        self.fc = nn.Sequential(nn.Linear(32*4*4,10),nn.Softmax(dim=0))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.block1(x)
        x = self.pool2d(x)
        x = self.block2(x)
        x = x.view(x.shape[0],-1)
        out = self.fc(x)
        return out
def main():
    model = Net()
    model.apply(init_weights)
    #method1 to initialize weights
    model.init_weights()
    #method2 to initialize weights
if __name__ == 'main':
    main()

#Look for more methods to initial your network
# https://pytorch.org/docs/stable/nn.html#torch-nn-init 
        
