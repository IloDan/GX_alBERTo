import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math
from pytorch_model_summary import summary


from src.config import (MAX_LEN, DROPOUT, DROPOUT_PE, DROPOUT_FC, MOD, center,
                        D_MODEL, N_HEAD, DIM_FEEDFORWARD, DEVICE, MASK,
                        NUM_ENCODER_LAYERS, OUTPUT_DIM, VOCAB_SIZE, FC_DIM, ATT_MASK, BATCH, REG_TOKEN)

class Embedding(nn.Module):
    def __init__(self, vocab_size= VOCAB_SIZE, embed_dim= D_MODEL, mask_embedding= MASK):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.mask_embedding = mask_embedding
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Initialization for nn.Embedding
        # self._init_embedding_weights()

    # def _init_embedding_weights(self):
    #     """
    #     Initialize embedding weights with uniform distribution in range [-0.05, 0.05].
    #     """
    #     if isinstance(self.embed, nn.Embedding):
    #         init_range = 0.05
    #         init.uniform_(self.embed.weight.data, -init_range, init_range)
        

    def forward(self, seq, met=None):
        if met is not None:
            return self._forward_with_met(seq, met)
        else:
            return self._forward_no_met(seq)

    def _forward_with_met(self, seq, met):
        if self.mask_embedding is not False:
            mask = (seq != self.mask_embedding).type_as(seq)
            mask = mask.unsqueeze(1).transpose(1,2)
        met_index = torch.full(met.shape, 5, dtype=torch.long).to(seq.device)

        seq = self.embed(seq)
        
        emb_met = self.embed(met_index)
        met = met.unsqueeze(-1)
        met = met*emb_met
        out = seq + met
        if self.mask_embedding is not False:
            return out * mask
        else:    
            return out
        
    def _forward_no_met(self, seq):
        if self.mask_embedding is not False:
            mask = (seq != self.mask_embedding).type_as(seq)
            mask = mask.unsqueeze(1).transpose(1,2)
        seq = self.embed(seq) 
        if self.mask_embedding is not False:
            return seq * mask #torch.Size([32, seq_len, 128]) * torch.Size([32, seq_len, 1])
        else:    
            return seq
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print('pe-size',pe.size())
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('x-size',x.size())
        # print('x-size0',x.size(0))
        # print('x-size1',x.size(1))
        x = x + self.pe[:, :x.size(1)]
        # print('x-size+pe',x.size())
        return self.dropout(x)
    
class Add_REG(nn.Module):
    def __init__(self, embed_dim, rate=0.01):
        super(Add_REG, self).__init__()
        REG_tok = torch.unsqueeze(torch.arange(1), 0)
        self.register_buffer('REG_tok', REG_tok)
        self.reg_emb = nn.Embedding(1, embed_dim)
        self.dropout = nn.Dropout(rate)

    def get_REG(self):
        return self.reg_emb(self.REG_tok)

    def forward(self, x, mask=None):
        REG = self.get_REG()
        REG = self.dropout(REG)
        REG = REG.expand(x.size(0), -1, -1)
        concat = torch.cat([REG, x], dim=1)
        if mask is not None:
            REG_mask = self.REG_tok.expand(x.size(0), -1).float()
            mask = torch.cat([REG_mask, mask], dim=1)
            return concat, mask
        return concat
    

def initialize_weights(*models): # model un oggetto con nn.MOdule
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

class multimod_alBERTo(nn.Module):
    def __init__(self):
        super(multimod_alBERTo, self).__init__()

        self.embedding = Embedding(vocab_size=VOCAB_SIZE, embed_dim=D_MODEL)
        
        # Convolutional layers
        self.conv1= nn.Sequential(
            nn.Conv1d(D_MODEL, D_MODEL, kernel_size=6, stride=1, padding='same'), 
            nn.ReLU(), 
            nn.Conv1d(D_MODEL, D_MODEL, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(D_MODEL, D_MODEL, kernel_size=6, stride=1, padding='same'), 
            nn.ReLU(),
            nn.Conv1d(D_MODEL, D_MODEL, kernel_size=1),
            nn.ReLU()
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(D_MODEL, D_MODEL, kernel_size=6, stride=1, padding='same'),
        #     nn.ReLU(),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(D_MODEL, D_MODEL, kernel_size=9, stride=1, padding='same'),
        #     nn.ReLU(),
        # )
        # self.fc = nn.Linear(2 * D_MODEL, D_MODEL) 

        #average pooling
        self.avgpool1d = nn.AvgPool1d(kernel_size=128, stride=128)
        self.batchnorm = nn.BatchNorm1d(D_MODEL, eps=1e-03)
        self.pooler = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.Tanh()
        )
        self.pos = PositionalEncoding(D_MODEL, MAX_LEN, DROPOUT_PE)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, 
                                                   dim_feedforward=DIM_FEEDFORWARD, 
                                                   dropout=DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        
        if REG_TOKEN:
            self.add_reg = Add_REG(D_MODEL)
        else:
            self.global_avg_pooling = nn.AdaptiveAvgPool1d(OUTPUT_DIM) 
        # MLP
        if MOD == 'metsum': 
            self.fc_block = nn.Sequential(
                nn.Linear(D_MODEL+1, FC_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(FC_DIM, FC_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(FC_DIM, OUTPUT_DIM),
            )
        else :
            self.fc_block = nn.Sequential(
                nn.Linear(D_MODEL, FC_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(FC_DIM, FC_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(FC_DIM, OUTPUT_DIM),
            )

     # Initialize parameters
        initialize_weights(self) 
        print(summary(self))

    def forward(self, src, met=None):
        

        if MASK:
            mask = src.detach()                 # N, L
            mask = src==MASK
            mask = F.max_pool1d(mask.float(), 128, 128)
        if MOD == 'met':
            src = self.embedding(src,met)       # N, L, C
        elif MOD == 'metsum':
            src = self.embedding(src)           # N, L, C
        else:
            raise ValueError("Invalid value for 'MOD'")        
        #transpose per convoluzione 1D
        src = src.transpose(2, 1)               # N, C, L
        # convoluzione 1D
        #src = self.conv1d(src)
        src1 = self.conv1(src)
        src2 = self.conv2(src)
        src12 = src1 + src2
        # src12 = torch.cat((src1, src2), dim=1)
        # src12 = src12.transpose(2, 1)
        # src12 = F.relu(self.fc(src12))
        # src = src.transpose(2, 1)
        src = src12 + src
        # average pooling
        # skip = skip.transpose(2, 1)
        #src = self.avgpool1d(src)
        src = self.avgpool1d(src)
        src = self.batchnorm(src)
        src = src.transpose(2, 1)                # N, L, C
    

        #src = self.pos(src)
        src = self.pos(src)
        #attention mask
        if ATT_MASK:
            mask = self.prepare_attention_mask(src)
            src, mask = self.add_reg(src, mask)
            encoded_features = self.transformer_encoder(src, mask)   
        else:
            src = self.add_reg(src)
            encoded_features = self.transformer_encoder(src)
        # encoded_features = encoded_features.transpose(1,2)
        # pooled_output = self.global_avg_pooling(encoded_features)
        # pooled_output = pooled_output.transpose(1,2)
        pooled_output = self.pooler(encoded_features[:, 0])
        if MOD == 'metsum':
            #somma dei valori di met tra center-400 e center
            metsum = torch.sum(met[:,center-400:center], dim=1)
            metsum = metsum.unsqueeze(1).unsqueeze(-1)
            pooled_output = torch.cat((pooled_output, metsum), dim=-1)
        pooled_output = pooled_output.squeeze(1)
        regression_output = self.fc_block(pooled_output)
        return regression_output.squeeze()