import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# from World_Model.Demo_model import Demo_WorldModel
# from Density_model.Demo_model import Demo_EnergyModel
from Energy_environment_models import TransformerDecoderModel
from Energy_environment_models import Demo_EnergyModel

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, initial_states, actions, next_states):
        self.buffer.append((initial_states, actions, next_states))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def select_action_energy_model(energy_model, state, action_dim,seq_len, num_samples=100, action_min=-1, action_max=1):
    with torch.no_grad():
        batch_size= state.shape[0]
        
        # Generate random action samples
        
        action_samples = (torch.rand(num_samples, seq_len, action_dim) * (action_max - action_min)) + action_min
        
        # Repeat the state for each action sample
        state_repeated = state.repeat(num_samples, 1, 1)
        
        # Reshape for the energy model
        # state_flat = state_repeated.view(-1, state.shape[-1])
        # action_flat = action_samples.view(-1, action_dim)
        #print(jd)
        # Compute energies
        energies = energy_model(state_repeated, action_samples)
        
        energies = torch.mean(energies, dim=(-2, -1))
    
        # Select the action with the highest energy
        best_action_idx = torch.argmax(energies)
    
        best_actions = action_samples[best_action_idx,:,:].unsqueeze(0)
        
        
        return best_actions
def run_energy_experiment(energy_model,  retrain_interval, state_dim, action_dim, replay_buffer,
                          replay_buffer_size=1000, learning_rate=1e-3, batch_size=128, energy_margin=0, num_epochs=5):
    # replay_buffer = ReplayBuffer(replay_buffer_size)
    energy_optimizer = optim.Adam(energy_model.parameters(), lr=learning_rate)
    action_min = torch.full((action_dim,), -1.0)
    action_max = torch.full((action_dim,), 1.0)

    
    for epoch in range(num_epochs):
                total_loss = 0
                # print(len(replay_buffer) , batch_size,"######e")
                # print(jklda)
                num_batches = len(replay_buffer) // batch_size#+1
                # print(num_batches)
                # print(jd)
                energy_loss=0
                for _ in range(num_batches):
                    batch = replay_buffer.sample(batch_size)
                    initial_states, actions, next_states = zip(*batch)
                    
                    initial_states = torch.cat(initial_states, dim=0)
                    actions = torch.cat(actions, dim=0)
                    #print(initial_states.shape,actions.shape,"####")
                    #print(jd)
                    # print(initial_states.shape,"####")
                    real_state_energy = energy_model(initial_states, actions)
                    neg_actions = torch.randn_like(actions)
                    unreal_state_energy = energy_model(initial_states, neg_actions)
                    
                    energy_loss += torch.relu(real_state_energy - unreal_state_energy + energy_margin).mean()
                    
                energy_loss=energy_loss/num_batches
                energy_optimizer.zero_grad()
                energy_loss.backward()
                energy_optimizer.step()
                
                total_loss += energy_loss.item()
                
                avg_loss = total_loss / num_batches
                print(f" Energy Model - Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
    
    return energy_model

def run_environment_experiment(world_model, energy_model, replay_buffer, num_steps, state_dim, action_dim, 
                               learning_rate=1e-3, batch_size=32, num_epochs=5):
    env_optimizer = optim.Adam(world_model.parameters(), lr=learning_rate)

    for step in range(num_steps):
        # Sample data from the replay buffer
        batch = replay_buffer.sample(batch_size)
        initial_states, actions, next_states = zip(*batch)
        
        # Convert to tensors
        initial_states = torch.cat(initial_states, dim=0)
        actions = torch.cat(actions, dim=0)
        next_states = torch.cat(next_states, dim=0)
        
        # Train the world model
        epoch_losses = []
        for epoch in range(num_epochs):
            predicted_states = world_model(initial_states, actions)
            env_loss = nn.MSELoss()(predicted_states, next_states)
            
            env_optimizer.zero_grad()
            env_loss.backward()
            env_optimizer.step()
            
            epoch_losses.append(env_loss.item())
        
        # Print average MSE after each epoch
        avg_mse = sum(epoch_losses) / num_epochs
        print(f"Averaged MSE {avg_mse:.4f}")
    
    return world_model
# Example usage
state_dim = 10
action_dim = 6
sequence_length = 20
batch_size = 1

# energy_model = Demo_EnergyModel(state_dim, action_dim)
energy_model = TransformerDecoderModel(state_dim=state_dim, action_dim=action_dim, 
                                      output_dim= 1)
world_model = TransformerDecoderModel(state_dim=state_dim, action_dim=action_dim, 
                                      output_dim= state_dim)


# Run energy model experiment
initial_states = torch.randn(batch_size, 1, state_dim)
actions = torch.randn(batch_size, sequence_length, action_dim)
next_states = torch.randn(batch_size, sequence_length, state_dim)#initial next state for intialization of the buffer

# Push data to replay buffer
replay_buffer = ReplayBuffer(1000)
replay_buffer.push(initial_states, actions, next_states)

trained_energy_model = run_energy_experiment(energy_model, num_epochs=10, retrain_interval=100, 
                                             state_dim=state_dim, action_dim=action_dim, 
                                             batch_size=batch_size,
                                             replay_buffer=replay_buffer)
terminal_state = next_states[:,-1,:].unsqueeze(0)

print(terminal_state.shape,"#####")
# print(jd)
action_chosen = select_action_energy_model(energy_model, terminal_state, action_dim, seq_len=20,num_samples=100, action_min=-1, action_max=1)
# print(action_chosen.shape," Actions chosen")
# print(jd)
actual_experiment_values = torch.randn(batch_size, sequence_length, state_dim)
### Push the actual values here.

replay_buffer.push(terminal_state, action_chosen, actual_experiment_values)

# Run environment model experiment
trained_world_model = run_environment_experiment(world_model, trained_energy_model,replay_buffer=replay_buffer, 
                                                 num_steps=100, state_dim=state_dim, 
                                                 action_dim=action_dim, batch_size=batch_size)