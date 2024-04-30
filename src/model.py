import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math
from pytorch_model_summary import summary
import importlib
import sys
import os

try:
    from src.config import (MAX_LEN, DROPOUT, DROPOUT_PE, DROPOUT_FC, MOD, center,
                        D_MODEL, N_HEAD, DIM_FEEDFORWARD, DEVICE, MASK,
                        NUM_ENCODER_LAYERS, PATIENCE, VOCAB_SIZE, FC_DIM, ATT_MASK, BATCH, REG_TOKEN)
except:
    from config import (MAX_LEN, DROPOUT, DROPOUT_PE, DROPOUT_FC, MOD, center,
                        D_MODEL, N_HEAD, DIM_FEEDFORWARD, DEVICE, MASK,
                        NUM_ENCODER_LAYERS,PATIENCE, VOCAB_SIZE, FC_DIM, ATT_MASK, BATCH, REG_TOKEN)

class Embedding(nn.Module):
    ''' 
    Embedding layer for input sequences.
    It support also a multimodality approach, adding to the input seq embedding a embedding for a vector of scalar value of the same length of the sequence.(met)-> see forward_met method
    Args:
        - vocab_size: size of vocabulary
        - embed_dim: dimension of embeddings
    '''
    def __init__(self, vocab_size= VOCAB_SIZE, embed_dim= D_MODEL):
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, seq, met=None):
        if met is not None:
            return self._forward_with_met(seq, met)
        else:
            return self._forward_no_met(seq)

    def _forward_with_met(self, seq, met):
        '''
        provide a way to add a vector of scalar values (met) to the input sequence.
        '''
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
    '''Positional encoding for the transformer.
    Args:
        - d_model: int, dimension of the model
        - max_len: int, maximum length of the input sequence
        - dropout: float, dropout rate
    '''
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
    
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    '''Custom Transformer Encoder Layer, provide a method for getting the attention weights from the self-attention layer.
    Args:
    - d_model: int, dimension of the model
    - nhead: int, number of heads in the multiheadattention models
    - dim_feedforward: int, dimension of the feedforward network model
    - dropout: float, dropout value
    - activation: str, the activation function of intermediate layer, relu or gelu -> Default: relu
    - batch_first: bool, whether the input and output tensors are batch first -> Default: True
    '''

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="relu", batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.batch_first = batch_first  # Store the batch_first attribute
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.attn_weights = None  

    def get_attn_weights(self):
        '''Returns the mean attention weights over the batch dimension'''
        mean_attn_weights = torch.mean(self.attn_weights, dim=0)
        return mean_attn_weights
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):

        src2, self.attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask, average_attn_weights = False)
        # print('self_att in custom encoder',self.attn_weights.size())
        # print('get_attn_weights mean on bath dim', self.get_attn_weights().size())
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
    
# class TransformerEncoder(nn.Module):
#     def __init__(self, num_layers, **block_args):
#         super().__init__()
#         self.layers = nn.ModuleList([CustomTransformerEncoderLayer(**block_args) for _ in range(num_layers)])

#     def forward(self, x, mask=None):
#         for layer in self.layers:
#             x = layer(x, mask=mask)
#         return x

#     def get_attention_maps(self, x, mask=None):
#         attention_maps = []
#         for layer in self.layers:
#             _, attn_map = layer.self_attn(x, return_attention=True)
#             attention_maps.append(attn_map)
#             x = layer(x)
#         return attention_maps

    
class Add_REG(nn.Module):
    '''Add a register token to the input sequence.
        It is used to provide a learnable token rappresenting the register of the sequence.
        If the input is a tensor of shape (N, L, C), the output will be a tensor of shape (N, L+1, C).
        It is used as input for the regressor fc_block, to get the final output.
    Args:
    - embed_dim: int, dimension of the model
    - rate: float, dropout rate
    '''
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
    '''
    Initialize the weights of the model.
    Args:   
    - models: list of nn.Module(layers of the models) to be initialized
    '''
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
    def __init__(self, max_len= MAX_LEN, vocab_size = VOCAB_SIZE, d_model = D_MODEL, output_dim = 1,
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
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
        #                                           nhead=self.num_heads, 
        #                                           dim_feedforward=dim_feedforward, 
        #                                           dropout=dropout, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
        #                                                  num_layers=num_encoder_layers)
        
        #with custom encoder layer
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=d_model,
                                          nhead=self.num_heads,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)
            for _ in range(num_encoder_layers)
        ])
        # self.trasformer_encoder = TransformerEncoder(num_encoder_layers, d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)

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
        print(summary(self, (torch.randint(0, VOCAB_SIZE, (BATCH, MAX_LEN)))))


    def forward(self, src, met=None):
        met = met #messo per un bug con summary s elo tolgo non stampa il modello
        if MASK:
            mask = src.detach()                 # N, L
            mask = src==MASK
            mask = F.max_pool1d(mask.float(), 128, 128)
        if MOD == 'met':
            src = self.embedding(src,met=met)       # N, L, C
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
        all_attn_weights = []
        if ATT_MASK:
            mask = self.prepare_attention_mask(src)
            src, mask = self.add_reg(src, mask)
            try:
                for layer in self.encoder_layers:
                    src= layer(src, mask)
                    attn_weights = layer.get_attn_weights()
                    all_attn_weights.append(attn_weights)
                encoded_features = src
                # all_attn_weights = torch.stack(all_attn_weights)
                # print('all_att_weights size:', all_attn_weights.size())
            except:
                encoded_features = self.transformer_encoder(src, mask)
        else:
            src = self.add_reg(src)
            # encoded_features = self.transformer_encoder(src) 
            try:
                for layer in self.encoder_layers:
                    src= layer(src)
                    attn_weights = layer.get_attn_weights()
                    all_attn_weights.append(attn_weights)
                encoded_features = src
                _attn_weights = torch.stack(all_attn_weights)
                # print('all_att_weights size:', _attn_weights.size())
            except:
                encoded_features = self.transformer_encoder(src)#usage of nn.TransformerEncoder e nn.TransformerEncoderLayer
                # encoded_features = self.trasformer_encoder(src)
                # all_attn_weights = self.trasformer_encoder.get_attention_maps(src)

            
        
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
        return regression_output.squeeze(), all_attn_weights

import matplotlib.pyplot as plt
import numpy as np
def plot_attention_maps(attn_maps, dir = None, epoch = None,  batch =None, input_data = None):
    '''
    Plot the attention maps for each head in each layer of the transformer encoder.

    Args:
        
    - attn_maps: list of torch.Tensor, attention maps from the model (is the argument passed to the function to plot the attention maps)
        - each attn_maps are a mean over the batch dimension computed in the CustomTransformerEncoderLayer
    - dir: str, directory where to save the plot -> Default: None (save in the current directory)
    - epoch: int, epoch number to save the plot -> Default: None
    - batch: int, batch number to save the plot -> Default: None
    - input_data: torch.Tensor, input data to the model if you want to plot the input data -> Default: None

    '''
    if input_data is not None:
        input_data = input_data.detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0].shape[-1])
    
    attn_maps = [m.detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 60 
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    plt.rcParams.update({'font.size': 40})  # Aumenta la dimensione del font

    # Calculate ticks: Include start, end, and every 20th index
    ticks = np.arange(0, seq_len, 20).tolist()
    if seq_len - 1 not in ticks:
        ticks.append(seq_len - 1)
    # Adjust the ax array to be a list of lists for uniformity in handling
    if num_layers == 1:
        ax = [ax]  # Make it a list of lists even when there is one layer
    if num_heads == 1:
        ax = [[a] for a in ax]  # Each row contains only one ax in a list if there is one head

    for row in range(num_layers):
        for column in range(num_heads):
            im = ax[row][column].imshow(attn_maps[row][column], origin="lower", cmap='viridis', vmin=0)
            #ax[row][column].set_xticks(list(range(seq_len))) 
            #ax[row][column].set_xticklabels(input_data.tolist(), rotation=90)  # Rotate labels if needed
            #ax[row][column].set_yticks(list(range(seq_len)))
            #ax[row][column].set_yticklabels(input_data.tolist())

            ax[row][column].set_xticks(ticks)           
            ax[row][column].set_xticklabels([input_data[t] for t in ticks], fontsize=80)  # Rotate labels if needed
            ax[row][column].set_yticks(ticks)
            ax[row][column].set_yticklabels([input_data[t] for t in ticks], fontsize=80)
            ax[row][column].set_title("Layer %i, Head %i" % (row + 1, column + 1), fontsize=120)
            # Create a color bar for each subplot
            cbar = fig.colorbar(im, ax=ax[row][column], orientation='vertical', shrink=0.5, aspect=20)
            cbar.ax.tick_params(labelsize=60)
            cbar.set_label('Attention Score', fontsize=80)
    plt.tight_layout()  # Apply tight layout to reduce spacing
    fig.subplots_adjust(hspace=1, wspace=1) 
    if dir is not None and epoch is not None and batch is not None:
        plt.savefig(os.path.join(dir, f'attention_maps_epoch{epoch}_batch_{batch}.png'))
    else :
        plt.savefig('attention_maps.png')
    plt.close()

if __name__=="__main__":
    seq_len = 2**13
    model = multimod_alBERTo()

    input = torch.randint(0, 5, (4, seq_len))
    output, att_weights = model(input)
    # crea nella directory corrente la cartella attn_plots se non esiste gia
    if not os.path.exists('attn_plots'):
        os.makedirs('attn_plots')
    print('output:', output, 'with shape:', output.size())
    print('attention weights:', type(att_weights), len(att_weights))
    plot_attention_maps(attn_maps=att_weights, dir='attn_plots', epoch=1, batch=1)

