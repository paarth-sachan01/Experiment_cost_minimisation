import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class TransformerDecoderModel(nn.Module):
    def __init__(self, state_dim, action_dim, output_dim, nhead=4, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        
        self.pos_encoder = nn.Embedding(1000, self.input_dim)  # Learnable position embeddings
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(self.input_dim, output_dim)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, initial_states, actions):
        batch_size, seq_length, _ = actions.shape
        
        # Combine initial states and actions
        initial_states_expanded = initial_states.repeat(1, seq_length, 1)
        #print(initial_states_expanded.shape,"#1")
        combined_input = torch.cat((initial_states_expanded, actions), dim=2)
        #print(combined_input.shape,"#2")
        # Add positional encoding
        positions = torch.arange(0, seq_length).unsqueeze(0).repeat(batch_size, 1).to(combined_input.device)
        combined_input += self.pos_encoder(positions)
        # print(positions.shape,self.pos_encoder(positions).shape,"#3")
        # print(combined_input.shape,"#$$!#")
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(seq_length).to(combined_input.device)
        
        # Decoder forward pass
        #print(combined_input.shape,initial_states_expanded.shape,tgt_mask.shape,"5")
        output = self.transformer_decoder(
            tgt=combined_input,
            memory=combined_input,  # Use combined_input as both tgt and memory
            tgt_mask=tgt_mask
        )
        output = output.view(-1, self.input_dim)
        #print(output.shape,"$$$$$$$$$")
        # Apply final linear layer
        output = self.fc(output)

        #print(output.shape,"$$$$$$$$$")
        output = output.reshape(batch_size,seq_length, -1)
        #print(output.shape,"33333$$$$$$$$$")
        #print(jd,"######")
        
        return output
    
class Demo_EnergyModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Demo_EnergyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)