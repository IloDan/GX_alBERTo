import torch
import torch.nn as nn

class ConcatCLS(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.01):
        super().__init__()
        CLS_tok = torch.unsqueeze(torch.arange(1), 0)
        self.register_buffer('CLS_tok', CLS_tok)
        self.CLS_embedding = nn.Embedding(1, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def get_CLS(self):
        return self.CLS_embedding(self.CLS_tok)          

    def forward(self, inputs, mask=None):
        batch_size = inputs.size(dim=0)
        CLS = self.get_CLS()
        CLS = self.dropout(CLS)
        CLS = CLS.repeat(batch_size, 1, 1)
        inputs = torch.cat((CLS,inputs), dim=1)
        if mask is not None:
            CLS_mask = self.CLS_tok.repeat(batch_size, 1).float()
            mask = torch.cat((CLS_mask,mask), dim=1)
            return inputs, mask
        return inputs
    
if __name__=="__main__":
    x = torch.rand((2,3,128))
    y = torch.randint(0,5,(2,3))
    CCLS = ConcatCLS(128)
    CCLS.train()
    out, mask = CCLS(x,y)
    print(out.shape, mask.shape)
