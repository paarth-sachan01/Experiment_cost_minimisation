import torch
import torch.nn as nn

# Placeholder for your existing model
class Demo_YourModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Demo_YourModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Density model placeholder
class Demo_DensityModel(nn.Module):
    def __init__(self, input_dim):
        super(Demo_DensityModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Actor placeholder
class Demo_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Demo_Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


# Main exploration system
class ExplorationSystem:
    def __init__(self, state_dim, action_dim, model_output_dim,
                 world_model =None, density_model= None, actor=None,
                 explore_algo= None):
        self.model = Demo_YourModel(state_dim, model_output_dim)
        self.density_model = Demo_DensityModel(model_output_dim)
        self.actor = Demo_Actor(state_dim + model_output_dim, action_dim)

    def density_based_exploration(self, states):
        # Step 1 & 2: Input states into the model
        model_outputs = self.model(states)

        # Step 3: Density model evaluates the outputs
        densities = self.density_model(model_outputs)

        # Combine states and model outputs for the actor
        actor_inputs = torch.cat([states, model_outputs], dim=1)

        # Step 5 & 6: Actor chooses the best input
        actions = self.actor(actor_inputs)

        # Select the action with the lowest density
        best_action_idx = torch.argmin(densities)
        best_action = actions[best_action_idx]

        return best_action, densities

# Demo usage
def demo():
    state_dim = 10
    action_dim = 5
    model_output_dim = 20
    batch_size = 32

    system = ExplorationSystem(state_dim, action_dim, model_output_dim)
    
    # Generate random states for demonstration
    states = torch.randn(batch_size, state_dim)

    best_action, densities = system.density_based_exploration(states)

    print("Best action:", best_action)
    print("Densities shape:", densities.shape)

if __name__ == "__main__":
    demo()