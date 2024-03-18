import torch.nn as nn
import torch
import math
from src.config import (MAX_LEN, DROPOUT, DROPOUT_PE, DROPOUT_FC, 
                        D_MODEL, N_HEAD, DIM_FEEDFORWARD, DEVICE, MASK,
                        NUM_ENCODER_LAYERS, OUTPUT_DIM, KERNEL_CONV1D, 
                        STRIDE_CONV1D, POOLING_OUTPUT, VOCAB_SIZE, FC_DIM)


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
        

    def forward(self, seq, met=None):
        if met is not None:
            return self._forward_with_met(seq, met)
        else:
            return self._forward_no_met(seq)

    def _forward_with_met(self, seq, met):
        if self.mask_embedding is not False:
            mask = (seq != self.mask_embedding).type_as(seq)
            
        met_index = torch.full(met.shape, 5, dtype=torch.long).to(DEVICE)

        seq = self.embed(seq)
        
        emb_met = self.embed(met_index)
        met = met.unsqueeze(-1)
        # print(emb_met.shape)
        # print(met.shape)
        met = met*emb_met
        out = seq + met
        if self.mask_embedding is not False:
            mask = (seq != self.mask_embedding).float()
            mask_embedded = mask.unsqueeze(-1)
            return out * mask_embedded
        else:    
            return out
        
    def _forward_no_met(self, seq):
        if self.mask_embedding is not False:
            mask = (seq != MASK).type_as(seq)
        seq = self.embed(seq)
        if self.mask_embedding is not False:
            mask = (seq != MASK).float()
            mask_embedded = mask.unsqueeze(-1)
            return seq * mask_embedded
        else:    
            return seq
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class multimod_alBERTo(nn.Module):
    def __init__(self):
        super(multimod_alBERTo, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, 
                                                   dim_feedforward=DIM_FEEDFORWARD, 
                                                   dropout=DROPOUT, batch_first=True)
     
        #convoluzione 1D
        self.conv1d = nn.Conv1d(in_channels=D_MODEL, out_channels=D_MODEL, kernel_size=KERNEL_CONV1D, stride=STRIDE_CONV1D, padding=1)
        #average pooling
        self.avgpool1d = nn.AdaptiveAvgPool1d(POOLING_OUTPUT)

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(OUTPUT_DIM)
        # Transformer
        self.embedding = Embedding(vocab_size=VOCAB_SIZE, embed_dim=D_MODEL)
        self.pos = PositionalEncoding(D_MODEL, MAX_LEN, DROPOUT_PE)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        # MLP
        self.fc_block = nn.Sequential(
            nn.Linear(D_MODEL, FC_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT_FC),
            nn.Linear(FC_DIM, OUTPUT_DIM),
        )

    def forward(self, src):
        src, mask = self.embedding(src)
        #transpose per convoluzione 1D
        src = src.transpose(2, 1)
        #convoluzione 1D
        src = self.conv1d(src)
        #average pooling
        src = self.avgpool1d(src)
        src = src.transpose(2, 1)
        src = self.pos(src)
        encoded_features = self.transformer_encoder(src)
        # # Prende solo l'output dell'ultimo token2 per la regressione
        encoded_features = encoded_features.transpose(1,2)
        # #print(encoded_features.shape)
        pooled_output = self.global_avg_pooling(encoded_features)
        # print(pooled_output.shape)
        pooled_output = pooled_output.transpose(1,2)
        regression_output = self.fc_block(pooled_output)
        return regression_output.squeeze()