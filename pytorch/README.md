# Pytorch Learning Notes

===
variables.py -- What is PyTorch? from Pytorch official tutorials Deeplearning with PyTorch:A 60 Minute Blitz

autograd.py Tries based on on th auto grad part, there are some questions about y.backward()

network.py Learn to construct networks, only finish the network initialization, forward and backward. Still have some questions.

All codes are from 

<https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>

Deep Learning with PyTorch: A 60 Minute Blitz

Author: Soumith Chintala

---

+ build_net.py  
  should work for Pytorch >=0.4.0
  + Network build tricks
        + nn.Sequential
        + Weight Initialization
        + basic block used
+ fine_tune.py
  + loss functions implemented by pytorch
  + optimizers and finetune(set different lr for different part)
  + to do:
    + self-defined loss function
    + examples of pytorch loss functions
+ recurrent_nets.py
  should work for Pytorch >=0.4.0
  + examples of RNN and LSTM
  + to do:
    + example of GRU
    + example of cells

All codes reffering Pytorch official doc
<https://pytorch.org/docs/stable/nn.html>
Most codes work for Pytorch >= 0.4.0 which is used when I start to learn Pytorch
Some codes only work for Pytorch>=1.0.0 which is my currently used version.

Pytroch env:
conda env base: python 3.7.1 and pytorch 0.4.1
conda env torch: python 3.6.8 and pytorch 1.1.0

To do list:

+ Spatial Transformation Network
+ Some useful functions
  + save checkpoint
  + load checkpoint
  + update model weight partialy
+ Dataset and Dataloader
+ GPU and Multi-gpu trianing
  + torch.cuda related
+ Pytorch and visdom
+ Optimizing training...

Welcome to share any other pytorch tricks to improve coding skills together.

If there is any fault please point it through Issue and I will update codes ASAP.

Hope this repo can help you.
