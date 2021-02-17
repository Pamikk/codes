import torch
#import torch.nn as nn
import torch.optim as optim

from build_net import Net
def adjust_learning_rate(cfg,optimizer,epoch):
    #adjust learning rate depending on epoch
    if (epoch+1) in cfg.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= cfg.lr_decays
    return optimizer.state_dict()['param_groups'][0]['lr']
def set_lr(named_params,lr,weight_decay):   
    params =[]
    for key,value in (dict(named_params).items()):
        if 'fc' in key:
            #adjust lr depending on block name
            params.append({'params':[value],'lr':lr*0.01})
        else:
            params.append({'params':[value],'lr':lr})
    optimizer = torch.optim.Adam(params,weight_decay = weight_decay)
    return optimizer
model = Net()
inp = torch.randn(3,3,32,32)
#b,c,h,w
out = model(inp)
print(out.shape)
target = torch.rand(3,10)
optimizer = set_lr(model.named_parameters(),0.001,0.0005)
criterion1 = nn.MSELoss(size_average=False)
criterion2 = nn.MSELoss(size_average=True)
loss1 = criterion1(out,target)
loss2 = criterion2(out,target)
# Loss general parameters--size_average
#     True/False, deprecated,get warnings in 0.4.0, replace by reduction
#     ignore when reduce is False
#     Default: True, average/sum over every element
criterion3 = nn.MSELoss(reduce=False)
loss3 = criterion3(out,target)
# Loss general parameters--reduce
#     True/False, deprecated,get warnings in 0.4.0, replace by reduction
#     ignore when reduce is False
#     Default: True, average/sum over every element
print(loss1,loss2,loss3)
print(loss2/loss1)
print(loss3.sum())
print(loss3.mean())
print(nn.MSELoss(reduction='none')(out,target))
print(nn.MSELoss(reduction='sum')(out,target))
print(nn.MSELoss()(out,target))
#loss general paramters--reduction
# 'none','mean','sum'
# same as loss3,loss1,loss2
# work for 1.1.0
# Pytorch 0.4.0 has no keyword 'mean',but default is mean...
# so just leave it blank...

# Othere useful loss function
# CTC loss(only avalible for Pytorch>=1.0)
criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
T = 50      # Input sequence length
C = 20      # Number of classes (excluding blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch
S_min = 10  # Minimum target length, for demonstration purposes
# Initialize random batch of input vectors, for *size = (T,N,C)
log_probs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# Initialize random batch of targets (0 = blank, 1:C+1 = classes)
target = torch.randint(low=1, high=C+1, size=(N, S), dtype=torch.long)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
loss = criterion(log_probs, target, input_lengths, target_lengths)
loss.backward()
#example from torch.nn docs
#be careful for line66: log_probs should be logarithmized, obtained from log_softmax
#                       shape of log_probs should be(T,N,C)
# target:(N,S) or (sum(target_lengths),)
# taget_lenths(N,) 
# input_lenths(N,)
#



