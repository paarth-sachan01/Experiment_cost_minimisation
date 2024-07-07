import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class TransformerDecoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=8, num_layers=6):
        super(TransformerDecoderModel, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, tgt, memory):
        output = self.transformer_decoder(tgt, memory)
        return self.fc(output)