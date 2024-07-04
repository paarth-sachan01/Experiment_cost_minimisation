import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(WorldModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class EnergyModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(EnergyModel, self).__init__()
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

def select_action_energy_model(energy_model, state, action_dim, num_samples=100):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).repeat(num_samples, 1)
        action_samples = torch.randn(num_samples, action_dim)
        energies = energy_model(state_tensor, action_samples)
        best_action_idx = torch.argmax(energies)
    return action_samples[best_action_idx].numpy()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward):
        self.buffer.append((state, action, next_state, reward))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def retrain_models(world_model, energy_model, replay_buffer, batch_size, learning_rate):
    world_optimizer = optim.Adam(world_model.parameters(), lr=learning_rate)
    energy_optimizer = optim.Adam(energy_model.parameters(), lr=learning_rate)
    
    if len(replay_buffer) < batch_size:
        return
    
    batch = replay_buffer.sample(batch_size)
    states, actions, next_states, _ = zip(*batch)
    
    # Retrain World Model
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    next_states = torch.FloatTensor(next_states)
    
    predicted_next_states = world_model(states, actions)
    world_loss = nn.MSELoss()(predicted_next_states, next_states)
    
    world_optimizer.zero_grad()
    world_loss.backward()
    world_optimizer.step()
    
    # Retrain Energy Model
    pos_energy = energy_model(states, actions)
    neg_actions = torch.FloatTensor(np.random.randn(*actions.shape))
    neg_energy = energy_model(states, neg_actions)
    
    energy_loss = (pos_energy - neg_energy).mean()
    
    energy_optimizer.zero_grad()
    energy_loss.backward()
    energy_optimizer.step()

def run_experiment(world_model, energy_model, num_steps, retrain_interval, state_dim, action_dim):
    state = np.random.randn(state_dim)  # Random initial state
    replay_buffer = ReplayBuffer(10000)
    
    for step in range(num_steps):
        action = select_action_energy_model(energy_model, state, action_dim)
        next_state = world_model(torch.FloatTensor(state).unsqueeze(0), torch.FloatTensor(action).unsqueeze(0)).squeeze(0).detach().numpy()
        reward = np.random.rand()  # Placeholder reward
        replay_buffer.push(state, action, next_state, reward)
        
        if step % retrain_interval == 0 and len(replay_buffer) >= 128:
            retrain_models(world_model, energy_model, replay_buffer, batch_size=128, learning_rate=0.001)
            print(f"Step {step}: Retrained models.")
        
        state = next_state
        if step % 100 == 0:
            print(f"Step {step}: State: {state}, Action: {action}")
    
    return world_model, energy_model

# Example usage
state_dim = 10
action_dim = 5
world_model = WorldModel(state_dim, action_dim)
energy_model = EnergyModel(state_dim, action_dim)
running_world_model, running_energy_model = run_experiment(world_model, energy_model, num_steps=1000, retrain_interval=100, state_dim=state_dim, action_dim=action_dim)