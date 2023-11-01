import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.layer_norm1(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_mask)
        x = x + self.dropout(attn_out)

        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + self.dropout(x)

        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, n_layers, dropout, seq_len):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim

        self.num_heads = num_heads
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.pos_enc = PositionalEncoding(self.emb_dim, dropout, seq_len)
        self.dropout = nn.Dropout(dropout)

        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    self.emb_dim,
                    self.num_heads,
                    self.emb_dim,
                    dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, src, src_mask):
        x = self.embedding(src)
        x = self.pos_enc(x)
        for l in self.layers:
            x = l(x, src_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, enc_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.enc_dim = enc_dim
        self.num_heads = num_heads

        self.self_attn1 = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.self_attn2 = nn.MultiheadAttention(self.embed_dim, self.num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, trg_mask):
        x = self.layer_norm1(x)
        attn_out, _ = self.self_attn1(
            x,
            x,
            x,
            attn_mask=trg_mask,
        )
        x = x + self.dropout(attn_out)

        x = self.layer_norm2(x)
        attn_out, _ = self.self_attn2(
            x,
            enc_output,
            enc_output,
            key_padding_mask=src_mask,
        )
        x = x + self.dropout(attn_out)

        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = x + self.dropout(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        enc_dim,
        num_heads,
        n_layers,
        dropout,
        seq_len,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.pos_enc = PositionalEncoding(self.emb_dim, dropout, seq_len)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.out = nn.Linear(in_features=emb_dim, out_features=output_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    self.emb_dim,
                    self.enc_dim,
                    self.num_heads,
                    self.emb_dim,
                    dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.embedding(x)
        x = self.pos_enc(x)
        for l in self.layers:
            x = l(x, enc_out, src_mask, trg_mask)
        return self.softmax(self.out(x))


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device,
        src_pad_idx,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_idx = src_pad_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimention instead of zero
        src_len = src.shape[0]
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # first input to the decoder is the <sos> tokens
        sos = trg[0, :].unsqueeze(0)
        input = sos

        src_mask = (src != self.src_pad_idx).transpose(0, 1).to(self.device)
        enc_out = self.encoder(src, src_mask)

        for t in range(1, max_len):
            trg_mask = (
                torch.tril(torch.ones(t, t))
                .expand(batch_size * self.decoder.num_heads, t, t)
                .to(self.device)
            )
            outputs = self.decoder(input, enc_out, src_mask, trg_mask)
            if t == max_len - 1:
                outputs = torch.cat(
                    (nn.functional.one_hot(sos, trg_vocab_size), outputs), dim=0
                )
                break
            outputs = torch.argmax(outputs, dim=-1)
            outputs = torch.cat((sos, outputs), dim=0)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[: t + 1] if teacher_force else outputs

        return outputs.float()


# From pytorch doc
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
