import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from src.config import (MAX_LEN, DROPOUT, DROPOUT_PE, DROPOUT_FC, MOD, center,
                        D_MODEL, N_HEAD, DIM_FEEDFORWARD, DEVICE, MASK,
                        NUM_ENCODER_LAYERS, OUTPUT_DIM, KERNEL_CONV1D, 
                        STRIDE_CONV1D, POOLING_OUTPUT, VOCAB_SIZE, FC_DIM)



# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim * embed_dim + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, embed_dim)
#         self.num_heads = num_heads
#         self.output_dim = output_dim

#     def forward(self, x):
#         # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
#         x = x.reshape(x.shape[0], x.shape[1] * x.shape[2]).permute(1, 0)  # N,H,W -> (HW),N

#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0).unsqueeze(-1)  # (HW+1)NC
#         pos = self.positional_embedding[:, None, :].to(x.dtype)
#         print('x', x.shape)
#         print('pos', pos.shape)
#         x = x +  pos # Add positional embedding
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0.,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )
#         print('x_final', x.shape)

#         return x[:self.output_dim]


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
        met = met*emb_met
        out = seq + met
        if self.mask_embedding is not False:
            mask = (seq != self.mask_embedding).float()
            return out * mask
        else:    
            return out
        
    def _forward_no_met(self, seq):
        if self.mask_embedding is not False:
            mask = (seq != MASK).type_as(seq)
        seq = self.embed(seq)
        if self.mask_embedding is not False:
            mask = (seq != MASK).float()
            return seq * mask
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

        # self.attention_pull = AttentionPool2d(MAX_LEN, D_MODEL, N_HEAD, POOLING_OUTPUT)	
        # self.global_attetion_pooling = AttentionPool2d(POOLING_OUTPUT, D_MODEL, N_HEAD, OUTPUT_DIM)
        # Transformer
        self.embedding = Embedding(vocab_size=VOCAB_SIZE, embed_dim=D_MODEL)
        self.pos = PositionalEncoding(D_MODEL, MAX_LEN, DROPOUT_PE)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        # MLP
        if MOD == 'metsum': 
            self.fc_block = nn.Sequential(
                nn.Linear(D_MODEL+1, FC_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(FC_DIM, OUTPUT_DIM),
            )
        else :
            self.fc_block = nn.Sequential(
                nn.Linear(D_MODEL, FC_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT_FC),
                nn.Linear(FC_DIM, OUTPUT_DIM),
            )




    def forward(self, src, met=None):
        if MOD == 'met':
            src = self.embedding(src,met)
        elif MOD == 'metsum':
            src = self.embedding(src)
        else:
            raise ValueError("Invalid value for 'MOD'")        
        #transpose per convoluzione 1D
        src = src.transpose(2, 1)
        # convoluzione 1D
        src = self.conv1d(src)
        # average pooling
        src = self.avgpool1d(src)
        # src = self.attention_pull(src)
        src = src.transpose(2, 1)
        src = self.pos(src)
        encoded_features = self.transformer_encoder(src)
        # # Prende solo l'output dell'ultimo token2 per la regressione
        encoded_features = encoded_features.transpose(1,2)
        # #print(encoded_features.shape)
        pooled_output = self.global_avg_pooling(encoded_features)
        # pooled_output = self.global_attetion_pooling(encoded_features)
        # print(pooled_output.shape)
        pooled_output = pooled_output.transpose(1,2)
        if MOD == 'metsum':
            #somma dei valori di met tra center-400 e center
            metsum = torch.sum(met[:,center-400:center], dim=1)
            metsum = metsum.unsqueeze(1).unsqueeze(-1)
            pooled_output = torch.cat((pooled_output, metsum), dim=-1)
        regression_output = self.fc_block(pooled_output)
        return regression_output.squeeze()