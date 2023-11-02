import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        n_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        max_len,
        device,
    ):
        super().__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_position_embedding = PositionalEncoding(d_model, dropout, max_len)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.trg_position_embedding = PositionalEncoding(d_model, dropout, max_len)
        self.device = device
        self.transformer = nn.Transformer(
            d_model,
            n_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        )
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def forward(self, src, trg):
        trg_seq_length, _ = trg.shape

        embed_src = self.src_position_embedding(self.src_word_embedding(src))
        embed_trg = self.trg_position_embedding(self.trg_word_embedding(trg))

        src_padding_mask = (
            (src.transpose(0, 1) == self.src_pad_idx).float().to(self.device)
        )

        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out


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
