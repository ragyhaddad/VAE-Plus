import torch 
from torch import nn 
import lightning as pl 


class Transformer(pl.LightningModule):
    def __init__(vocab, d_model=512, num_layers=6):
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embedding = nn.Embedding(vocab.vocab_size, d_model)
