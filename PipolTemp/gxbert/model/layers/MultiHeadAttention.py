import torch
import torch.nn as nn
import math
from torch.nn import functional as F

def attention(q, k, v, d_k, mask=None, dropout=None, masked_token=4, softplus=None, temperature=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) # math.sqrt(d_k) # softplus(temperature)
    if mask is not None:
        # mask = mask.unsqueeze(1)
        # scores = scores.masked_fill(mask == masked_token, -1e9)
        scores += (mask*-1e9)

    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output, scores

class TFMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = None, masked_token=4):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model
        self.h = heads
        self.masked_token = masked_token
        
        self.q_linear = nn.Linear(d_model, d_model*heads)
        self.v_linear = nn.Linear(d_model, d_model*heads)
        self.k_linear = nn.Linear(d_model, d_model*heads)
        if dropout != None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.out = nn.Linear(d_model*heads, d_model)
        self.temperature = nn.Parameter(torch.tensor([0]).float())
        self.softplus = nn.Softplus()
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        output, scores = attention(q, k, v, self.d_k, mask, self.dropout, self.masked_token, self.softplus, self.temperature)
        
        # concatenate heads and put through final linear layer
        concat = output.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model*self.h)
        
        output = self.out(concat)
    
        return output, scores.detach()

class TOMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = None, masked_token=4):
        super().__init__()
        
        self.d_model = d_model
        assert d_model % heads == 0 
        self.d_k = d_model // heads
        self.h = heads
        self.masked_token = masked_token
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        if dropout != None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.out = nn.Linear(d_model, d_model)
        self.temperature = nn.Parameter(torch.tensor([0]).float())
        self.softplus = nn.Softplus()
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
# calculate attention using function we will define next
        output, scores = attention(q, k, v, self.d_k, mask, self.dropout, self.masked_token, self.softplus, self.temperature)
        
        # concatenate heads and put through final linear layer
        concat = output.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output, scores.detach()        

if __name__=="__main__":
    x = torch.rand((2,3,128))
    y = torch.randint(0,5,(2,3))
    y = y.view(2, 1, 3, 1)
    y = y.repeat(1,4,1,3)
    y = y.transpose(-2,-1)
    # y = torch.matmul(y.transpose(1,2),y)
    mha = TFMultiHeadAttention(4, 128)
    out = mha(x,x,x,y)
    print(out.shape)
    total_params = sum(p.numel() for p in mha.parameters())
    print("Total Parameters:", total_params)