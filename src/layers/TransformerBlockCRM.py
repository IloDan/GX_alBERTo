import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, seq_len, rate=0.1, batch_first=True, mem_len=128, CLS=True):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim, batch_first=batch_first)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(),nn.Linear(ff_dim, embed_dim)
        )
       
        if CLS:
            self.CLS = 1
            CLS_tok = torch.unsqueeze(torch.arange(self.CLS), 0)
            self.register_buffer('CLS_tok', CLS_tok)
            self.CLS_embedding = nn.Embedding(self.CLS, embed_dim)
        else:
            self.CLS = 0

        self.layernorm1 = nn.LayerNorm([seq_len+mem_len+self.CLS, embed_dim], eps=1e-6)
        self.layernorm2 = nn.LayerNorm([seq_len+mem_len+self.CLS, embed_dim], eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.seq_len = seq_len
        self.mem_len = mem_len
       
        if mem_len>0:
            mem_pos = torch.unsqueeze(torch.arange(mem_len), 0)
            self.register_buffer('mem_pos', mem_pos)
            self.embedding = nn.Embedding(mem_len, embed_dim)

    def get_CLS(self):
        return self.CLS_embedding(self.CLS_tok)  

    def get_memory(self,):
        return self.embedding(self.mem_pos)        

    def forward(self, inputs, masked=None):
        batch_size = inputs.size(dim=0)
        if self.CLS == 1:
            CLS = self.get_CLS()
            CLS = CLS.repeat(batch_size, 1, 1)
            inputs = torch.cat((CLS,inputs), dim=1)
        if self.mem_len > 0:
            memory = self.get_memory()
            memory = memory.repeat(batch_size, 1, 1)
            inputs = torch.cat((inputs,memory), dim=1)
        if masked is not None:
            attn_output, _ = self.att(inputs, inputs, inputs, key_padding_mask=masked, average_attn_weights=False)
        else:
            attn_output, _ = self.att(inputs, inputs, inputs, average_attn_weights=False)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        output = self.layernorm2(out1 + ffn_output)
        return output[:, :self.seq_len+self.CLS, :]