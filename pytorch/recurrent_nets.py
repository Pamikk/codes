import torch
import torch.nn as nn
# Reference: https://pytorch.org/docs/stable/nn.html#recurrent-layers
input_size = 5        #number of expected features in the input x
hidden_size = 6       #number of features in hidden vector
num_layers = 2        #default:1
nonlinearity = 'tanh' #'tanh'|'relu' default:'tanh'
bias = True           # if use bias weights,default True
batch_first = False   # if batch the first dim,default: False
dropout = 0           # dropout probability, default:0
bidirectional = True  # become bidirectional RNN
rnn = nn.RNN(input_size,hidden_size,num_layers,nonlinearity = nonlinearity,
             bias=bias,batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
# forward
inp = torch.randn(10,3,5) #seq_len,batch_num,input_size
h0 = torch.randn(4,3,6)   #num_layers*num_directions,batch_size,hidden_size, default:zero if not provided
output,h_n = rnn(inp,h0)
print(output.shape) #seq,batch,num_directions*hs
print(h_n.shape)    #num_layers*num_directions,bs,hs
print(list(rnn.state_dict().keys()))

# For LSTM, nonlinerity is 'tanh', so no nonlinearity parameter
# output will be one more cn(cell state)
lstm = nn.LSTM(input_size,hidden_size,num_layers,bias=bias,
             batch_first=batch_first,dropout=dropout,bidirectional=bidirectional)
output,h_n_c_n = lstm(inp)# default h0,c0, 0 
h_n,c_n = h_n_c_n
print(output.shape)
print(h_n.shape,c_n.shape)
print(list(lstm.state_dict().keys()))
 