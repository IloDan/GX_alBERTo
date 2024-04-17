import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
import math

from configu import get_config

from configu import (MAX_LEN, MOD, center,
                     N_HEAD, DEVICE, MASK,
                     OUTPUT_DIM, VOCAB_SIZE, ATT_MASK, D_MODEL)

config = get_config()


class Embedding(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=D_MODEL, mask_embedding=MASK):
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
            mask = mask.unsqueeze(1).transpose(1, 2)
        met_index = torch.full(met.shape, 5, dtype=torch.long).to(DEVICE)

        seq = self.embed(seq)

        emb_met = self.embed(met_index)
        met = met.unsqueeze(-1)
        met = met * emb_met
        out = seq + met
        if self.mask_embedding is not False:
            return out * mask
        else:
            return out

    def _forward_no_met(self, seq):
        if self.mask_embedding is not False:
            mask = (seq != MASK).type_as(seq)
            mask = mask.unsqueeze(1).transpose(1, 2)
        seq = self.embed(seq)
        if self.mask_embedding is not False:
            return seq * mask  # torch.Size([32, seq_len, 128]) * torch.Size([32, seq_len, 1])
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
    

class PrepareAttentionMask(nn.Module):
    def __init__(self, add_reg, pool_size):
        super(PrepareAttentionMask, self).__init__()
        self.add_reg = add_reg
        self.pool_size = pool_size
        self.num_heads = N_HEAD

    def forward(self, x):
        x = 1 - x
        if x.size(2) >= self.pool_size:  # Controlla la dimensione dell'input
            x = F.max_pool1d(x, self.pool_size)
        if self.add_reg:
            zeros = torch.zeros((x.size(0), 1, 1), device=x.device)
            x = torch.cat([zeros, x], dim=1)
        x = 1 - x
        x = torch.bmm(x, x.transpose(1, 2))
        # Ripeti la maschera di attenzione per ogni testina
        x = x.repeat(self.num_heads, 1, 1)
        return x


class multimod_alBERTo(nn.Module):
    def __init__(self, config):
        super(multimod_alBERTo, self).__init__()

        self.embedding = Embedding(vocab_size=VOCAB_SIZE, embed_dim=D_MODEL)
        self.pooler = nn.Linear(D_MODEL, D_MODEL)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD,
                                                   dim_feedforward=config['DIM_FEEDFORWARD'],
                                                   dropout=config['DROPOUT'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['NUM_ENCODER_LAYERS'])

        # convoluzione 1D
        self.conv1 = nn.Conv1d(D_MODEL, D_MODEL, kernel_size=6, stride=1, padding='same')
        self.conv2 = nn.Conv1d(D_MODEL, D_MODEL, kernel_size=9, stride=1, padding='same')
        self.fc = nn.Linear(2 * D_MODEL, D_MODEL)

        # self.conv1d = nn.Conv1d(in_channels=D_MODEL, out_channels=D_MODEL, kernel_size=KERNEL_CONV1D, stride=STRIDE_CONV1D, padding=1)

        # average pooling
        self.avgpool1d = nn.AvgPool1d(kernel_size=128, stride=128)
        self.batchnorm = nn.BatchNorm1d(D_MODEL)

        self.pos = PositionalEncoding(D_MODEL, MAX_LEN, config['DROPOUT_PE'])

        self.prepare_attention_mask = PrepareAttentionMask(add_reg=False, pool_size=128)

        # self.add_reg = Add_REG(D_MODEL)

        # self.avgpool1d = nn.AdaptiveAvgPool1d(POOLING_OUTPUT)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(OUTPUT_DIM)

        # MLP
        if MOD == 'metsum':
            self.fc_block = nn.Sequential(
                nn.Linear(D_MODEL + 1, config['FC_DIM']),
                nn.ReLU(),
                nn.Dropout(config['DROPOUT_FC']),
                nn.Linear(config['FC_DIM'], config['FC_DIM']),
                nn.GELU(),
                nn.Dropout(config['DROPOUT_FC']),
                nn.Linear(config['FC_DIM'], OUTPUT_DIM),
            )
        else:
            self.fc_block = nn.Sequential(
                nn.Linear(D_MODEL, config['FC_DIM']),
                nn.ReLU(),
                nn.Dropout(config['DROPOUT_FC']),
                nn.Linear(config['FC_DIM'], config['FC_DIM']),
                nn.GELU(),
                nn.Dropout(config['DROPOUT_FC']),
                nn.Linear(config['FC_DIM'], OUTPUT_DIM),
            )

    def forward(self, src, met=None):
        if MOD == 'met':
            src = self.embedding(src, met)
        elif MOD == 'metsum':
            src = self.embedding(src)
        else:
            raise ValueError("Invalid value for 'MOD'")
            # transpose per convoluzione 1D
        src = src.transpose(2, 1)
        # convoluzione 1D
        # src = self.conv1d(src)
        src1 = F.relu(self.conv1(src))
        src2 = F.relu(self.conv2(src))
        src12 = torch.cat((src1, src2), dim=1)
        src12 = src12.transpose(2, 1)
        src12 = F.relu(self.fc(src12))
        del src1, src2
        src = src.transpose(2, 1)
        skip = src12 + src
        del src12, src
        # average pooling
        skip = skip.transpose(2, 1)
        # src = self.avgpool1d(src)
        x = self.avgpool1d(skip)
        del skip
        x = self.batchnorm(x)
        x = x.transpose(2, 1)

        # src = self.pos(src)
        x = self.pos(x)

        # attention mask
        if ATT_MASK:
            att_mask = self.prepare_attention_mask(x)
            encoded_features = self.transformer_encoder(x, att_mask)
        # x = self.add_reg(x)
        else:
            encoded_features = self.transformer_encoder(x)
        # x = self.add_reg(x)

        encoded_features = encoded_features.transpose(1, 2)
        pooled_output = self.global_avg_pooling(encoded_features)
        pooled_output = pooled_output.transpose(1, 2)
        # pooled_output = self.pooler(encoded_features[:, 0])
        if MOD == 'metsum':
            # somma dei valori di met tra center-400 e center
            metsum = torch.sum(met[:, center - 400:center], dim=1)
            metsum = metsum.unsqueeze(1).unsqueeze(-1)
            pooled_output = torch.cat((pooled_output, metsum), dim=-1)
        pooled_output = pooled_output.squeeze(1)
        pooled_output = torch.tanh(pooled_output)
        regression_output = self.fc_block(pooled_output)
        return regression_output.squeeze()