import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, dropout, forward_expansion):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(12, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embed_size, 1)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        src = self.embedding(src).permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        if tgt is not None:
            tgt = self.embedding(tgt).permute(1, 0, 2)
        else:
            tgt = src
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output.mean(dim=0))
