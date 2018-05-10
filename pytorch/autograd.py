from __future__ import print_function
import torch
import numpy as np 
x=torch.ones(3,3,requires_grad=True)
y=x+2
z=y*y*3
out=z.mean()
print(out)
print(y.grad_fn)
a=torch.randn(2,2)
a=(a*3)/(a-1)
a.requires_grad_(True)
#a.requires_grad(True) will cause TypeError:bool object is not callable
b=(a*a).sum()
#Gradients
out.backward()
print(x.grad) #d(out)/dx
#crazy things?
x=torch.randn(3,requires_grad=True)
y=x*2
while y.data.norm()<1000:
    y=y*2
print(y)
print(x.grad)
#y.backward() Runtime error... think caused by while???
gradients=torch.ones(3,dtype=torch.float)
y.backward(gradients)
print(x.grad)
gradients = torch.tensor([0.1,1,0.0001],dtype=torch.float)
y.backward(gradients)
print(x.grad)
print ((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)
    print(x.requires_grad)
print(x.requires_grad)
