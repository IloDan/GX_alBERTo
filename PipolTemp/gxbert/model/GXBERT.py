import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from layers.PositionalEncoding import PositionalEncoding
from layers.TransformerBlock import TransformerBlockTFP
from layers.ConcatCLS import ConcatCLS
from layers.BERTPooler import BERTPooler
from pytorch_model_summary import summary
# import logging

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                init_range=0.05
                init.uniform_(module.weight.data, -init_range, init_range)
            if isinstance(module, nn.Conv1d):
                init.xavier_normal_(module.weight) # xavier_uniform_
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0) 


class GXBERT(nn.Module):
    def __init__(self, seq_len=2**14, vocab_size=5, d_model=128, max_pool=128, num_heads=4, expand_dim=4, 
                    t_encoder_layers=1, mlp_neurons=128, output_neurons=1, CLS=True, mem_len=0, pooler="BERT", dropout_rate=0.1,
                    masked_token=4):
        super(GXBERT, self).__init__()
        self.masked_token = masked_token
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.conv_layers_1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=6, padding="same"),
            nn.ReLU(),
        )
        self.conv_layers_2 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=9, padding="same"),
            nn.ReLU(),
        )
        self.project_1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.ReLU()
        )
        self.project_2 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1),
            nn.ReLU()
        )
        self.CLS = CLS
        self.max_pool=max_pool
        self.num_heads=num_heads
        if self.CLS:
            self.concatCLS = ConcatCLS(embed_dim=d_model)
        self.avgpool = nn.AvgPool1d(kernel_size=max_pool)
        self.batchnorm = nn.BatchNorm1d(d_model, eps=1e-3, momentum=0.1, track_running_stats=True) # eps=0.001, momentum=0.01 # InstanceNorm1d # BatchNorm1d
        # self.batchnorm = EpsBatchNorm1d(d_model, eps=1e-3, momentum=0.1)
        self.max_len = seq_len//max_pool 
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.max_len)
        self.max_len += int(CLS)
        self.t_encoder_layers = nn.ModuleList([TransformerBlockTFP(embed_dim=d_model, num_heads=num_heads, 
                                            ff_dim=d_model*expand_dim, seq_len=seq_len//max_pool, mem_len=mem_len, CLS=self.CLS,
                                            rate=0.1, batch_first=True, masked_token=masked_token) for _ in range(t_encoder_layers)])
        if pooler=="BERT":
            self.pooling = BERTPooler(d_model)
        else:
            self.pooling = torch.nn.AdaptiveAvgPool1d(1)   
        
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, mlp_neurons),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_neurons, mlp_neurons),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_neurons, output_neurons),
        )

        # Initialize parameters
        initialize_weights(self) 


        print(summary(self, (torch.randint(0, 5, (16, seq_len)))))

    def forward(self, x):
        x = x.long()

        mask = x.detach()               # N, L
        mask = x==self.masked_token
        mask = mask.float()
        mask = F.max_pool1d(mask.float(), self.max_pool, self.max_pool)

        x = self.embedding(x)           # N, L, C
        x = torch.permute(x, (0, 2, 1)) # N, C, L
        skip = x                        # N, C, L
        x_1 = self.conv_layers_1(x)
        x_1 = self.project_1(x_1)

        x_2 = self.conv_layers_2(x)     # N, C, L
        x_2 = self.project_2(x_2)       # N, C, L

        x = x_1 + x_2                   # N, C, L

        x = x + skip                    # N, C, L
        x = self.avgpool(x)             # N, C, L
        x = self.batchnorm(x)           # N, C, L
        x = torch.permute(x, (0, 2, 1)) # N, L, C
        x = self.pos_encoder(x)         # N, L, C
        if self.CLS:
            x, mask = self.concatCLS(x, mask)       # N, L, C
        
        mask = mask==0
        mask = mask.float()
        mask = mask.view(mask.size(0), 1, self.max_len, 1)
        mask = mask.repeat(1,self.num_heads,1,1)
        mask = torch.matmul(mask, mask.transpose(-2,-1))
        mask = mask==0
        mask = mask.float()

        for t_encoder in self.t_encoder_layers:
            x, _ = t_encoder(x, mask)            # N, L, C
        x = self.pooling(x)             # N, 1, C
        x = x.view(x.size(0), -1)       # N, C
        x = self.fc_layers(x)           # N, 1
        return x

if __name__=="__main__":
    seq_len = 2**4
    model = GXBERT(seq_len=seq_len,expand_dim=4, mlp_neurons=256, max_pool=2)

    input = torch.randint(0, 5, (2, seq_len))
    output = model(input)
    print(output)

