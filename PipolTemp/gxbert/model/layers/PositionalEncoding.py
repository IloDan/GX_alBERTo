import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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



if __name__=="__main__":
    pos_enc = PositionalEncoding(128, dropout=0)
    x = torch.zeros((1,64,128))
    x = pos_enc(x)
    x = x.reshape(64,128)
    # position = 64
    # d_model = 128
    # pos_encoding = positional_encoding(position, d_model)
    # x = pos_encoding
    # x = x.reshape(64,128)
    plt.imshow(x, cmap='RdBu', vmin=-1, vmax=1)
    plt.title("Pos_enc")
    plt.xlabel("Depth")
    plt.ylabel("Position")
    plt.savefig("pos_enc2.png")
