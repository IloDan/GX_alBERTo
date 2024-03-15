import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        # Utilizziamo nn.Embedding per l'embedding dei token
        self.embedding = nn.Embedding(4, embed_dim)  # 4 token nel vocabolario

    def forward(self, inputs, masked=None):
        # Applichiamo l'embedding ai token di input
        embedded_input = self.embedding(inputs)
        # Eseguiamo l'attenzione multi-testa
        if masked is not None:
            attn_output, att_scores = self.att(embedded_input, embedded_input, embedded_input, key_padding_mask=masked)
        else:
            attn_output, att_scores = self.att(embedded_input, embedded_input, embedded_input)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(embedded_input + attn_output)
        # Applichiamo la rete feedforward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        # Aggiungiamo la connessione residua e normalizzazione layer
        return self.layernorm2(out1 + ffn_output), att_scores


class FNETBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, rate=0.1, compression=False, shift_freq=False, asymmetric=False):
        super(FNETBlock, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.compression = compression
        self.shift_freq = shift_freq
        self.asymmetric = asymmetric

    def forward(self, inputs):
        # Calcola la FFT (Trasformata di Fourier)
        fft2D = torch.fft.fftn(inputs.type(torch.complex64))  # Converti in complesso
        out1 = self.layernorm1(inputs + fft2D.real)  # Usa la parte reale della FFT

        if self.shift_freq:
            if self.compression is not False:
                cut = int((out1.shape[-1] * self.compression) // 2)  # Modifica la dimensione
                out1 = out1[:, :, cut:-cut]  # Modifica la dimensione
        else:
            if self.compression is not False:
                cut = int((out1.shape[-1] * (1 - self.compression)) // 2)  # Modifica la dimensione
                outA = out1[:, :, :cut + 1]  # Modifica la dimensione
                outB = out1[:, :, -cut - 1:]  # Modifica la dimensione
                out1 = torch.cat([outA, outB], -1)  # Modifica la dimensione

        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class projTransformer(nn.Module):
    def __init__(self,
                embed_dim=32,
                num_heads=4,
                ff_dim=64,
                compression=False,
                shift_freq=False,
                asymmetric=False):

        super(projTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.compression = compression
        self.shift_freq = shift_freq
        self.asymmetric = asymmetric

        self._build_model()

    def _build_model(self):
        # TransformerBlocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
            for _ in range(3)  # Adjust number of blocks as needed
        ])

        # FNETBlock
        self.fnet_block = FNETBlock(self.embed_dim, self.ff_dim, compression=self.compression, shift_freq=self.shift_freq, asymmetric=self.asymmetric)

    def forward(self, inputs):
        for transformer_block in self.transformer_blocks:
            inputs, _ = transformer_block(inputs)

        inputs = self.fnet_block(inputs)

        return inputs
