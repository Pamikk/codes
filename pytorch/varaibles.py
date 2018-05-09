from __future__ import print_function
import torch
import numpy as np
#dtype=torch.float
#x=torch.empty(5,3)
#initialize 5 rows 3 cols random matrix(2-d tensor)

x=torch.rand(5,3)
#initialize with normal

#x=torch.zeros(5,3,dtype=torch.long)
#initialize with all zero

#x=torch.tensor([5.5,3])
#initialize with constant

#x=x.new_ones(5,3,dtype=torch.double)
#new_*methods take in sizes
#print(x)

#x=torch.randn_like(x,dtype=torch.float) #override dtype
#print(x)

print(x.size()) # output is a tuple

y=torch.rand(5,3)

print(x+y)
#result=torch.empty(5,3)
#print(torch.add(x,y,out=result))
#y.add_(x)
#x.copy_(y),x.t_()(transpose = x=x.t()) will change x
#reshape torch.view()
x=torch.rand(4,4)
print(x.view(-1,2,2))#-1 means inferredn from other dimensions
y=x.numpy()
print(y)
x.add_(1)
print(x)
print(y)# dynmically 

a=np.ones(3)
print(a)
b=torch.from_numpy(a)
a=a+1 #not influence b???
np.add(a,1,out=a)#not influence b???
print(b)
if torch.cuda.is_available():
    device=torch.device("cuda")
    print(1)