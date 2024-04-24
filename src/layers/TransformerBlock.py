import torch
import torch.nn as nn
from .MultiHeadAttention import TFMultiHeadAttention
from .MultiHeadAttention import TOMultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, rate=0.1, batch_first=True, mem_len=128, CLS=True):
        super(TransformerBlock, self).__init__()
        self.att = TOMultiHeadAttention(heads=num_heads, d_model=embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),nn.Linear(ff_dim, embed_dim)
        )
       
        if CLS:
            self.CLS = 1
        else:
            self.CLS = 0

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.seq_len = seq_len
        self.mem_len = mem_len
       
        if mem_len>0:
            mem_pos = torch.unsqueeze(torch.arange(mem_len), 0)
            self.register_buffer('mem_pos', mem_pos)
            self.embedding = nn.Embedding(mem_len, embed_dim)

    def get_memory(self,):
        return self.embedding(self.mem_pos)        

    def forward(self, inputs, mask=None):
        batch_size = inputs.size(dim=0)
        if self.mem_len > 0:
            memory = self.get_memory()
            memory = memory.repeat(batch_size, 1, 1)
            inputs = torch.cat((inputs,memory), dim=1)
        if mask is not None:
            attn_output, scores = self.att(inputs, inputs, inputs, mask)
        else:
            attn_output, scores = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        output = self.layernorm2(out1 + ffn_output)
        return output[:, :self.seq_len+self.CLS, :], scores
    
class TransformerBlockTFP(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, rate=0.1, batch_first=True, mem_len=128, CLS=True, masked_token=4):
        super(TransformerBlockTFP, self).__init__()
        self.att = TFMultiHeadAttention(heads=num_heads, d_model=embed_dim, masked_token=masked_token)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),nn.Linear(ff_dim, embed_dim)
        )
       
        if CLS:
            self.CLS = 1
        else:
            self.CLS = 0

        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.seq_len = seq_len
        self.mem_len = mem_len
       
        if mem_len>0:
            mem_pos = torch.unsqueeze(torch.arange(mem_len), 0)
            self.register_buffer('mem_pos', mem_pos)
            self.embedding = nn.Embedding(mem_len, embed_dim)

    def get_memory(self,):
        return self.embedding(self.mem_pos)        

    def forward(self, inputs, mask=None):
        batch_size = inputs.size(dim=0)
        if self.mem_len > 0:
            memory = self.get_memory()
            memory = memory.repeat(batch_size, 1, 1)
            inputs = torch.cat((inputs,memory), dim=1)
        if mask is not None:
            attn_output, scores = self.att(inputs, inputs, inputs, mask)
        else:
            attn_output, scores = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        output = self.layernorm2(out1 + ffn_output)
        return output[:, :self.seq_len+self.CLS, :], scores    