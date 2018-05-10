import torch
import torch.nn as nn
import torch.nn.functional as F

#Define Neural Network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        # input feature size,output feature size, bias=T/F
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #x is input feature of the layer
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# backward function is automatically defined for using autograd
net = Net()
print(net)
#Output:
'''Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)'''
params=list(net.parameters())
print(len(params))
print(params[0],params[0].size())
inp=torch.randn(1,1,32,32)
#batch,channel,h,w
inp.requires_grad_(True)
out=net(inp)
print(out)
net.zero_grad() #zero the gradiet buffers of all parameters
out.backward(torch.randn(1,10))# random gradients
#torch.nn only supports mini-batches and not a single sample
#single sample needs to use input.unsqueeze(0) fake batch dimension
#using unsqueeze does not work, need to try more and learn more about unsqueeze
print(inp.grad) 