import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math
# from pytorch_model_summary import summary

from config import (MAX_LEN, DROPOUT, DROPOUT_PE, DROPOUT_FC, MOD, center,
                        D_MODEL, N_HEAD, DIM_FEEDFORWARD, MASK,
                        NUM_ENCODER_LAYERS, OUTPUT_DIM, VOCAB_SIZE, FC_DIM, ATT_MASK, BATCH, REG_TOKEN)

class Embedding(nn.Module):
    def __init__(self, vocab_size= VOCAB_SIZE, embed_dim= D_MODEL):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, seq, met=None):
        if met is not None:
            return self._forward_with_met(seq, met)
        else:
            return self._forward_no_met(seq)

    def _forward_with_met(self, seq, met):
        met_index = torch.full(met.shape, 5, dtype=torch.long).to(seq.device)
        seq = self.embed(seq)
        emb_met = self.embed(met_index)
        met = met.unsqueeze(-1)
        met = met*emb_met
        out = seq + met
        return out
        
    def _forward_no_met(self, seq):
        seq = self.embed(seq)   
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
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
    def __init__(self, max_len= MAX_LEN, vocab_size = VOCAB_SIZE, d_model = D_MODEL, output_dim = OUTPUT_DIM,
                 dropout = DROPOUT, dropout_fc = DROPOUT_FC, dropout_pe = DROPOUT_PE,
                 n_heads = N_HEAD, dim_feedforward = DIM_FEEDFORWARD, masked_token = MASK,
                 num_encoder_layers = NUM_ENCODER_LAYERS, REG=REG_TOKEN, fc_dim = FC_DIM):
        super(multimod_alBERTo, self).__init__()
        
        self.masked_token = masked_token
        # Embedding
        self.embedding = Embedding(vocab_size=vocab_size, embed_dim=d_model)
        
        # Convolutional layers
        self.conv1= nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=6, stride=1, padding='same'), 
            nn.ReLU(), 
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=6, stride=1, padding='same'), 
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU()
        )
        self.REG=REG
        self.num_heads = n_heads
        if REG:
            self.add_reg = Add_REG(d_model)

        #average pooling
        self.avgpool1d = nn.AvgPool1d(kernel_size=128, stride=128)
        self.batchnorm = nn.BatchNorm1d(d_model, eps=1e-03, momentum=0.1, track_running_stats=True)

        self.max_len = max_len//128
        if REG:
            self.pooler = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh()
            )
        else:
            self.global_avg_pooling = nn.AdaptiveAvgPool1d(output_size=output_dim) 
        
        self.pos = PositionalEncoding(d_model, self.max_len, dropout_pe)
        self.max_len += int(REG)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        

        # MLP
        if MOD == 'metsum': 
            self.fc_block = nn.Sequential(
                nn.Linear(d_model+1, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout_fc),
                nn.Linear(fc_dim, fc_dim),
                nn.GELU(),
                nn.Dropout(dropout_fc),
                nn.Linear(fc_dim, output_dim),
            )
        else :
            self.fc_block = nn.Sequential(
                nn.Linear(d_model, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout_fc),
                nn.Linear(fc_dim, fc_dim),
                nn.GELU(),
                nn.Dropout(dropout_fc),
                nn.Linear(fc_dim, output_dim),
            )
            
     # Initialize parameters
        initialize_weights(self) 
        # print(summary(self, (torch.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN)))))


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
        src1 = self.conv1(src)
        src2 = self.conv2(src)
        src12 = src1 + src2

        #skip connection
        src = src12 + src
        # average pooling e batchnorm
        src = self.avgpool1d(src)
        src = self.batchnorm(src)
        src = src.transpose(2, 1)                # N, L, C
        #positional encoding
        src = self.pos(src)
        #attention mask
        if ATT_MASK:
            mask = self.prepare_attention_mask(src)
            src, mask = self.add_reg(src, mask)
            encoded_features = self.transformer_encoder(src, mask)   
        else:
            src = self.add_reg(src)
            encoded_features = self.transformer_encoder(src)

        if REG_TOKEN:
            pooled_output = self.pooler(encoded_features[:, 0])
        else:
            encoded_features = encoded_features.transpose(1,2)
            pooled_output = self.global_avg_pooling(encoded_features)
            pooled_output = pooled_output.transpose(1,2)
        
        
        if MOD == 'metsum':
            #somma dei valori di met tra center-400 e center
            metsum = torch.sum(met[:,center-400:center], dim=1)
            metsum = metsum.unsqueeze(-1)
            pooled_output = torch.cat((pooled_output, metsum), dim=-1)
        pooled_output = pooled_output.squeeze(1)
        regression_output = self.fc_block(pooled_output)
        return regression_output.squeeze()
    

if __name__=="__main__":
    seq_len = 2**11
    model = multimod_alBERTo()

    input = torch.randint(0, 5, (2, seq_len))
    output = model(input)
    print(output)

