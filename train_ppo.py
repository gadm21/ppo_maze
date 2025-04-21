"""
Trains a Proximal Policy Optimization (PPO) agent to navigate a simple grid world.

This script defines the PPO algorithm components, the neural network model
(actor-critic), a simple grid environment, and the main training loop.
The trained model is saved to 'ppo_model.pth'.
"""

import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical
from tqdm import tqdm  # Progress bar
import numpy as np
import matplotlib.pyplot as plt 
import os
import random
import gymnasium as gym # Import gymnasium
from gymnasium import spaces # Import spaces

def set_global_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(seed)

set_global_seeds(42)


# Define available training maps
TRAINING_MAPS = [
    os.path.join("maps", "train_map_1.txt"),
    os.path.join("maps", "train_map_2.txt"),
    os.path.join("maps", "train_map_3.txt"),
]


class GridEnv(gym.Env): # Inherit from gym.Env
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, map_file, max_steps=50):
        super().__init__() # Initialize the parent class
        self.max_steps = max_steps
        self.current_step = 0

        # --- Map Loading --- #
        self.map = self._load_map(map_file)
        self.height = self.map.shape[0]
        self.width = self.map.shape[1]
        self.wall_positions = set(tuple(pos) for pos in np.argwhere(self.map == 'W'))
        target_pos_list = np.argwhere(self.map == 'T')
        if len(target_pos_list) == 0:
            raise ValueError("Map must contain at least one target 'T'.")
        self.target_pos = tuple(target_pos_list[0]) # Use the first target found
        self.empty_positions = [tuple(pos) for pos in np.argwhere(self.map == 'E')]
        if not self.empty_positions:
             raise ValueError("Map must contain at least one empty 'E' cell for the agent to start.")
             
        # Add helper methods for reward calculation
        # Removed last_actions as anti-loop logic is not used

        # --- Action Space --- #
        # 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)

        # --- Observation Space --- #
        # observation: 1-channel image
        image_shape = (1, self.height, self.width) # Use actual height and width
        

        self.observation_space = spaces.Box(low=0, high=3, shape=image_shape, dtype=np.float32)
         
        # --- Agent State --- #
        self.agent_pos = self._get_random_start_pos() # Initial agent position

    def _load_map(self, map_file):
        """Loads the map from a text file."""
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Map file not found: {map_file}")
        with open(map_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Map file is empty: {map_file}")
        # Basic validation: ensure all lines have the same length
        map_width = len(lines[0])
        if any(len(line) != map_width for line in lines):
             raise ValueError("Map rows have inconsistent lengths.")
        return np.array([list(line) for line in lines])

    def reset(self):
        """Resets the environment, placing the agent at a random empty position."""
        # Select a random starting position from the available empty spots
        self.agent_pos = self._get_random_start_pos()
        self.current_step = 0
        return self._get_obs()

    def _get_random_start_pos(self):
        """Gets a random empty position from the map."""
        pos = random.choice(self.empty_positions)
        # Removed last_actions reset as anti-loop logic is not used
        return pos
        
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_obs(self):
        """Gets the current observation (image representation and position)."""
        # Create image based on actual height and width. 0: empty, 1: agent, 2: target, 3: wall
        image = np.zeros((1, self.height, self.width), dtype=np.float32)
        # Agent position: 1.0 at agent's position
        image[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        # Target position: 2.0 at target's position
        image[0, self.target_pos[0], self.target_pos[1]] = 2.0
        # Wall positions: 3.0 at wall positions]
        for r, c in self.wall_positions:
            image[0, r, c] = 3.0
        return image

    def step(self, action):
        """Executes one step in the environment.

        Args:
            action (int): The action to take (0: up, 1: down, 2: left, 3: right).

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Calculate potential new position
        new_pos = self.agent_pos
        if action == 0: new_pos = (new_pos[0] - 1, new_pos[1])  # Up
        elif action == 1: new_pos = (new_pos[0] + 1, new_pos[1])  # Down
        elif action == 2: new_pos = (new_pos[0], new_pos[1] - 1)  # Left
        elif action == 3: new_pos = (new_pos[0], new_pos[1] + 1)  # Right
        else:
            # add penalty for invalid action
            reward -= 1.0 # Penalty for invalid action
            new_pos = self.agent_pos

        reward = -min(self.height,self.width) # Basic time penalty per step
        done = False
        reason = ""

        # Distance-based reward shaping
        prev_dist = self._manhattan_distance(self.agent_pos, self.target_pos)
        # Check boundaries using height and width
        if not (0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width):
            reward -= self.height * self.width
            new_pos = self.agent_pos
            reason = "Boundary"
            done = True
        elif new_pos in self.wall_positions:
            reward -= self.height * self.width
            new_pos = self.agent_pos
            reason = "Wall"
            done = True

        # If move was valid (not boundary or wall), update agent position
        self.agent_pos = new_pos
        self.current_step += 1

        # Reward for moving closer to the target, penalty for moving away
        new_dist = self._manhattan_distance(self.agent_pos, self.target_pos)
        if new_dist < prev_dist:
            reward += new_dist**2
        elif new_dist > prev_dist:
            reward -= new_dist**2

        # Check if target reached
        if self.agent_pos == self.target_pos:
            reward += self.height * self.width * 2 # Reward for reaching target
            done = True
            reason = "Target"
        
        reward = float(reward)

        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            reason = "Max Steps"

        observation = self._get_obs()
        info = {"reason": reason}
        return observation, reward, done, info


class PolicyValueNet(nn.Module):
    """Combined policy and value network using CNN for image and FC for position."""
    def __init__(self, input_channels, num_actions, grid_height, grid_width):
        super(PolicyValueNet, self).__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Convolutional layers for the image part of the observation
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=3, stride=1, padding=1)
        self.conv1_relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
        self.conv2_relu = nn.ReLU()
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.conv3_relu = nn.ReLU()

        # Calculate the flattened size of the CNN output
        conv_output_size = self._get_conv_output_size(grid_height, grid_width)

        # Fully connected layers processing combined features
        self.fc1 = nn.Linear(conv_output_size, 56) # Adjusted input size
        self.fc1_relu = nn.ReLU()

        # Actor head (policy)
        self.actor = nn.Linear(56, num_actions)

        # Critic head (value)
        self.critic_linear = nn.Linear(56, 1)


    def _get_conv_output_size(self, height, width):
        """Helper function to calculate the output size of the conv layers."""
        # Simulate passing a dummy tensor through the conv layers
        # Input shape: (batch_size, channels, height, width)
        dummy_input = pt.zeros(1, self.conv1.in_channels, height, width)
        x = self.conv1(dummy_input)
        x = self.conv1_relu(x)
        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x = self.conv3_relu(x)
        # Return the total number of features after flattening
        return int(np.prod(x.size()[1:])) # Multiply dimensions C*H*W

    def forward(self, x):
        """Forward pass through the network."""
        image = x # x is image tensor
        # Ensure image is 4D (Batch, Channel, Height, Width)
        # CNN path
        x = self.conv1(image)
        x = self.conv1_relu(x)
        x = self.conv2(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x = self.conv3_relu(x)
        x = x.view(x.size(0), -1) # Flatten

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc1_relu(x)

        # Actor and critic heads
        action_logits = self.actor(x)
        value = self.critic_linear(x)

        return action_logits, value


# def ppo_update(model, optimizer, images, actions, log_probs_old, returns, advantages,
#                clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01,
#                epochs=4, mini_batch_size=16, device='cpu', grad_clip=0.5, log_losses=False):
#     """Performs the PPO update step.

#     Args:
#         model (PolicyValueNet): The model to update.
#         optimizer (torch.optim.Optimizer): The optimizer.
#         images (np.array): Batch of image observations.
#         actions (np.array): Batch of actions taken.
#         log_probs_old (np.array): Log probabilities of actions under the old policy.
#         returns (np.array): Batch of calculated returns (target for value function).
#         advantages (np.array): Batch of calculated advantages.
#         clip_param (float): PPO clipping parameter (epsilon).
#         value_loss_coef (float): Coefficient for the value loss term.
#         entropy_coef (float): Coefficient for the entropy bonus term.
#         epochs (int): Number of optimization epochs per PPO update.
#         mini_batch_size (int): Size of mini-batches for optimization.
#         device (torch.device): CPU or CUDA device.
#     """
#     # Convert numpy arrays to tensors
#     images_tensor = pt.FloatTensor(images).to(device)
#     actions_tensor = pt.LongTensor(actions).to(device)
#     log_probs_old_tensor = pt.FloatTensor(log_probs_old).to(device)
#     returns_tensor = pt.FloatTensor(returns).to(device)
#     advantages_tensor = pt.FloatTensor(advantages).to(device)

#     # Normalize advantages
#     adv_mean = advantages_tensor.mean()
#     adv_std = advantages_tensor.std(unbiased=False) if advantages_tensor.numel() > 1 else pt.tensor(1.0, device=advantages_tensor.device)
#     if adv_std.item() != 0:
#         advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8) # Avoid division by zero
#     else:
#         advantages_tensor = advantages_tensor - adv_mean

#     batch_size = images.shape[0]

#     # Optimization loop
#     for _ in range(epochs):
#         # Create mini-batches
#         indices = np.arange(batch_size)
#         np.random.shuffle(indices)
#         for start in range(0, batch_size, mini_batch_size):
#             end = start + mini_batch_size
#             mb_indices = indices[start:end]

#             # Get mini-batch data
#             mb_images = images_tensor[mb_indices]
#             mb_actions = actions_tensor[mb_indices]
#             mb_log_probs_old = log_probs_old_tensor[mb_indices]
#             mb_returns = returns_tensor[mb_indices]
#             mb_advantages = advantages_tensor[mb_indices]

#             # --- Calculate Loss --- #
#             # Get new policy distribution and value estimates from the model
#             observation = mb_images
#             action_logits, value_new = model(observation)
#             action_dist_new = Categorical(logits=action_logits)

#             # Calculate log probabilities of actions taken under the *new* policy
#             log_probs_new = action_dist_new.log_prob(mb_actions)

#             # Calculate the policy ratio (pi_new / pi_old)
#             ratio = pt.exp(log_probs_new - mb_log_probs_old)

#             # Calculate the clipped surrogate objective (Policy Loss)
#             surr1 = ratio * mb_advantages
#             surr2 = pt.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
#             policy_loss = -pt.min(surr1, surr2).mean()

#             # Calculate the value function loss (MSE)
#             value_loss = pt.nn.functional.mse_loss(value_new.squeeze(), mb_returns.squeeze()) # Ensure both tensors are squeezed

#             # Calculate the entropy bonus (encourages exploration)
#             entropy = action_dist_new.entropy().mean()

#             # Calculate the total loss
#             loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

#             # --- Optimization Step --- #
#             optimizer.zero_grad()
#             loss.backward()
#             # Gradient clipping
#             pt.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
#             optimizer.step()
#             if log_losses:
#                 print(f"Policy loss: {policy_loss.item():.4f}, Value loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")



def ppo_update(model, optimizer, images, actions, old_log_probs, advantages, returns, 
               clip_epsilon, policy_coef,entropy_coef, value_coef, epochs, batch_size, device):
    """
    Perform PPO update using collected trajectories.

    Args:
        model: PolicyValueNet model (outputs action logits and value).
        optimizer: Adam optimizer for model parameters.
        images (np.ndarray): Stored observations (shape: [num_steps, *obs_shape]).
        actions (np.ndarray): Stored actions (shape: [num_steps]).
        old_log_probs (np.ndarray): Log probabilities of actions (shape: [num_steps]).
        advantages (np.ndarray): GAE advantages (shape: [num_steps]).
        returns (np.ndarray): Discounted returns (shape: [num_steps]).
        clip_epsilon (float): PPO clipping parameter.
        policy_coef (float): Policy loss coefficient.
        entropy_coef (float): Entropy bonus coefficient.
        value_coef (float): Value loss coefficient.
        epochs (int): Number of optimization epochs.
        batch954_size (int): Mini-batch size.
        device: PyTorch device (cuda or cpu).

    Returns:
        tuple: Average policy loss, value loss, and entropy over all updates.
    """
    # Convert numpy arrays to PyTorch tensors
    images = pt.FloatTensor(images).to(device)
    actions = pt.LongTensor(actions).to(device)
    old_log_probs = pt.FloatTensor(old_log_probs).to(device)
    advantages = pt.FloatTensor(advantages).to(device)
    returns = pt.FloatTensor(returns).to(device)

    # Normalize advantages (already done in compute_gae_and_returns, but ensure consistency)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Get total number of samples
    num_samples = images.shape[0]
    indices = np.arange(num_samples)

    # Track losses for logging
    policy_losses = []
    value_losses = []
    entropies = []

    # Perform multiple epochs of optimization
    for _ in range(epochs):
        # Shuffle indices for mini-batch sampling
        np.random.shuffle(indices)
        
        # Process mini-batches
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # Extract mini-batch data
            batch_images = images[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Forward pass: compute new action logits and value
            action_logits, values = model(batch_images)
            action_dist = Categorical(logits=action_logits)
            
            # Compute new log probabilities and entropy
            batch_log_probs = action_dist.log_prob(batch_actions)
            entropy = action_dist.entropy().mean()

            # Compute policy loss (PPO clipped objective)
            ratios = pt.exp(batch_log_probs - batch_old_log_probs)
            surr1 = ratios * batch_advantages
            surr2 = pt.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
            policy_loss = -pt.min(surr1, surr2).mean()

            # Compute value loss (mean squared error)
            value_loss = pt.nn.functional.mse_loss(values.squeeze(-1), batch_returns)

            # Total loss
            total_loss = policy_coef * policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Store losses for logging
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())

    # Return average losses
    return (np.mean(policy_losses), np.mean(value_losses), np.mean(entropies))

# Training loop
def train():
    """
    Main training function for the PPO agent.

    Initializes the environment, model, and optimizer. Runs the training loop
    for a specified number of episodes, collecting trajectories, calculating
    returns and advantages, and updating the model using the ppo_update function.
    Saves the trained model at the end.
    """
    # Hyperparameters
    initial_learning_rate = 1e-4  # Lower learning rate for stability
    gamma = 0.95         # Discount factor (Higher values favor long-term rewards)
    gae_lambda = 0.9    # Lambda for GAE (Encourages long-term rewards. Lower values favor short-term rewards)
    clip_epsilon = 0.1   # PPO clip parameter (Controls the range of the policy update)
    entropy_coef = 0.05  # Entropy coefficient (encourage exploration, not too high)
    value_coef = 0.5     # Value loss coefficient (Controls the importance of value function)
    policy_coef = 1.0 - value_coef - entropy_coef    # Policy loss coefficient (Controls the importance of policy)
    epochs = 10          # PPO epochs per update
    batch_size = 164      # PPO batch size
    num_steps = 150     # Steps per trajectory collection (buffer size - larger for stability)
    episodes = 2000      # Total training episodes
    show_every = 20      # How often to print average reward
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # Initialize model based on the dimensions of the first map
    temp_env_for_dims = GridEnv(map_file=TRAINING_MAPS[0], max_steps=1000)
    initial_height = temp_env_for_dims.height
    initial_width = temp_env_for_dims.width 
    obs_space = temp_env_for_dims.observation_space # Get obs space structure
    num_actions = temp_env_for_dims.action_space.n
    del temp_env_for_dims # Don't need it anymore

    # Filter TRAINING_MAPS to only include maps with the initial dimensions
    print(f"Initializing model for maps with dimensions: {initial_height}x{initial_width}")
    filtered_training_maps = []
    for map_path in TRAINING_MAPS:
        try:
            temp_env = GridEnv(map_file=map_path, max_steps=1000)
            if temp_env.height == initial_height and temp_env.width == initial_width:
                filtered_training_maps.append(map_path)
            else:
                print(f"Skipping map {map_path} due to different dimensions ({temp_env.height}x{temp_env.width})")
            del temp_env
        except Exception as e:
            print(f"Error loading map {map_path}: {e}. Skipping.")

    if not filtered_training_maps:
        raise ValueError("No training maps found matching the dimensions of the first map.")

    print(f"Using {len(filtered_training_maps)} maps for training: {filtered_training_maps}")

    # Initialize the model using the determined dimensions
    input_channels = obs_space.shape[0]
    print(f"Input channels: {input_channels}")
    model = PolicyValueNet(
        input_channels= input_channels, 
        num_actions=num_actions,
        grid_height=initial_height, # Pass height
        grid_width=initial_width   # Pass width
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, eps=1e-5)

    # Storage for trajectories
    images_buf = np.zeros((num_steps, *obs_space.shape), dtype=np.float32)
    actions_buf = np.zeros(num_steps, dtype=np.int32)
    log_probs_buf = np.zeros(num_steps, dtype=np.float32)
    values_buf = np.zeros(num_steps, dtype=np.float32)
    rewards_buf = np.zeros(num_steps, dtype=np.float32)
    dones_buf = np.zeros(num_steps, dtype=bool) # Use standard bool

    total_rewards = []
    best_avg_reward = -float('inf') # Initialize best average reward

    # Main training loop
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        # --- Learning Rate Decay --- #
        frac = 1.0 - (episode - 1.0) / episodes
        current_lr = initial_learning_rate * frac
        optimizer.param_groups[0]["lr"] = current_lr # Update optimizer's LR

        # --- Select Map and Reset Environment for the Episode --- #
        current_map_file = random.choice(filtered_training_maps) # Use filtered list
        env = GridEnv(map_file=current_map_file, max_steps=num_steps)

        observation = env.reset()
        episode_reward = 0

      
        # --- Trajectory Collection --- #
        for t in range(num_steps):
            image_obs = observation
            image_tensor = pt.FloatTensor(image_obs).to(device) # Convert observation to tensor

            with pt.no_grad():
                # Add batch dimension for model inference
                image_tensor_batched = image_tensor.unsqueeze(0)
                action_logits, value = model(image_tensor_batched) # Get action logits and state value

            action_dist = Categorical(logits=action_logits) # Create action distribution
            action = action_dist.sample() # Sample action
            log_prob = action_dist.log_prob(action) # Compute log probability of action

            next_observation, reward, done, _ = env.step(action.item()) # Take action in environment

            # Store transition data
            images_buf[t] = image_obs
            actions_buf[t] = action.item() # Store action
            log_probs_buf[t] = log_prob.item() # Store log probability
            values_buf[t] = value.item() # Store state value
            rewards_buf[t] = reward # Store reward
            dones_buf[t] = done # Store done flag

            observation = next_observation
            episode_reward += reward

            if done:
                break

        # --- Calculate Advantages and Returns --- #
        # --- Calculate Advantages and Returns using simplified function --- #
        # Get the value of the last state for bootstrapping
        if not done:
            image_tensor = pt.FloatTensor(observation).to(device)
            with pt.no_grad():
                image_tensor_batched = image_tensor.unsqueeze(0)
                _, last_value = model(image_tensor_batched)
            last_value = last_value.item()
        else:
            last_value = 0.0

        advantages, returns = compute_gae_and_returns(
            rewards_buf, values_buf, dones_buf, last_value, gamma, gae_lambda
        )

        # --- Perform PPO Update --- #
        ppo_update(model, optimizer, images_buf, actions_buf, log_probs_buf, returns, advantages, clip_epsilon=clip_epsilon, policy_coef=policy_coef, value_coef=value_coef, 
                entropy_coef=entropy_coef, epochs=epochs, batch_size=batch_size, device=device)



        # Logging
        total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards[-100:]) # Calculate average reward over last 100 episodes

        # Log average reward every show_every episodes
        if (episode + 1) % show_every == 0:
            print(f"\nEpisode {episode + 1}/{episodes} | Avg Reward (last 100): {avg_reward:.2f} | Last Reward: {episode_reward:.2f}")

        # Save the best model based on average reward
        if avg_reward > best_avg_reward and len(total_rewards) >= 100: # Start saving after 100 episodes
            best_avg_reward = avg_reward
            best_model_state = model.state_dict().copy()

    # --- End of Training --- 

    # Save the best model if we found one, otherwise save the final model
    if best_model_state is not None:
        pt.save(best_model_state, "ppo_model.pth")
        print(f"\nBest model saved to ppo_model.pth (Avg Reward: {best_avg_reward:.2f})")
    else:
        pt.save(model.state_dict(), "ppo_model.pth")
        print("\nFinal model saved to ppo_model.pth")

def compute_gae_and_returns(rewards, values, dones, last_value, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    Args:
        rewards (np.ndarray): Rewards for each step.
        values (np.ndarray): Value estimates for each step.
        dones (np.ndarray): Done flags (bool) for each step.
        last_value (float): Value estimate for the last state.
        gamma (float): Discount factor.
        gae_lambda (float): GAE lambda parameter.
    Returns:
        advantages (np.ndarray): Advantage estimates (normalized).
        returns (np.ndarray): Return estimates.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
    returns = advantages + values
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns

# Entry point for the script
if __name__ == "__main__":



    print("Starting PPO training...")
    train() # Run the training function
    print("Training finished.")