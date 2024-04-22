import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np

def get_angles(pos, i, d_model, c1=10_000):
    angle_rates = 1 / torch.pow(c1, (2 * (i // 2)) / torch.tensor(d_model,dtype=torch.float32))
    return pos * angle_rates

def positional_encoding(position, d_model, c1=10_000):
    position = torch.arange(position).unsqueeze(1)
    div_term = get_angles(position, torch.arange(d_model), d_model, c1)
    # Apply sin to even indices in the array; 2i
    div_term[:, 0::2] = torch.sin(div_term[:, 0::2])
    # Apply cos to odd indices in the array; 2i+1
    div_term[:, 1::2] = torch.cos(div_term[:, 1::2])
    pos_encoding = div_term.unsqueeze(0)
    return pos_encoding.float()

class PositionalEncoding(nn.Module):
    """
    from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=1000, c1=10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = positional_encoding(max_len, d_model, c1)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        x = x*torch.sqrt(torch.tensor(self.d_model,dtype=torch.float32))
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#nostro positional encoding
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=1000, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)
    


# if __name__=="__main__":
  
#     # pos_enc = PositionalEncoding(128, dropout=0)
#     # x = torch.zeros((1,64,128))
#     # x = pos_enc(x)
#     # x = x.reshape(64,128)
  
#     position = 64
#     d_model = 128
#     pos_encoding = positional_encoding(position, d_model)
#     x = pos_encoding
#     x = x.reshape(64,128)
   
#     # plt.imshow(x, cmap='RdBu', vmin=-1, vmax=1)
#     # plt.title("Pos_enc")
#     # plt.xlabel("Depth")
#     # plt.ylabel("Position")
#     # plt.savefig("pos_enc_nos.png")

#     # Squeeze the tensor to remove dimensions of size 1, resulting in shape (32, 128)
#     pos_encoding = pos_encoding.squeeze()  # Rimuovere eventuali dimensioni inutili

#     # Plot configuration
#     plt.figure(figsize=(15, 6))
    
#     # Plot all sine components in one subplot
#     plt.subplot(1, 2, 1)  # One row, two columns, first plot
#     for i in range(3):  # Plot the first 5 sine components
#         plt.plot(pos_encoding[:, 2 * i].numpy(), label=f'Sine {i+1}')
#     plt.title('Sine Components')
#     plt.xlabel('Position')
#     plt.ylabel('pos_emb_odd')
#     plt.legend()
    
#     # Plot all cosine components in one subplot
#     plt.subplot(1, 2, 2)  # One row, two columns, second plot
#     for i in range(3):  # Plot the first 5 cosine components
#         plt.plot(pos_encoding[:, 2 * i + 1].numpy(), label=f'Cosine {i+1}', linestyle='--')
#     plt.title('Cosine Components')
#     plt.xlabel('Position')
#     plt.ylabel('pos_emb_even')
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig("pip_pos_enc_sinusoids.png")
