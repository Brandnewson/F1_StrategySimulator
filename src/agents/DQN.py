import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Optional

from base_agents import BaseAgent, DriverAction, RiskLevel
from feedback import create_driver_feedback


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a random batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class SimpleDQN(nn.Module):
    """Simple fully-connected DQN network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    """Agent that uses Deep Q-Network (DQN) to select actions.
    
    This agent learns from experience using Q-learning with experience replay
    and a target network. It can operate in training mode (with exploration)
    or evaluation mode (greedy action selection).
    """
    
    ACTION_SPACE = [
        (RiskLevel.CONSERVATIVE, False),
        (RiskLevel.CONSERVATIVE, True),
        (RiskLevel.NORMAL, False),
        (RiskLevel.NORMAL, True),
        (RiskLevel.AGGRESSIVE, False),
        (RiskLevel.AGGRESSIVE, True),
    ]
    
    def __init__(
        self,
        config: dict,
        state_dim: int,
        name: str = "DQNAgent",
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
    ):
        super().__init__(name=name)
        
        self.action_dim = len(self.ACTION_SPACE)
        self.state_dim = state_dim
        
        # Q-network and target network
        self.model = SimpleDQN(state_dim, self.action_dim, hidden_size)
        self.target_model = SimpleDQN(state_dim, self.action_dim, hidden_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # Training components
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        
        # Training state
        self.training_steps = 0
        self.episodes_completed = 0  # Track episode count for epsilon decay
        self.training_mode = config.get("agent_mode") == "training"
        
        # Store last state and action for learning
        self.last_state = None
        self.last_action_idx = None
        self.last_feedback = None
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: State vector
            explore: Whether to use epsilon-greedy (True) or greedy (False)
            
        Returns:
            Action index
        """
        if explore and self.training_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return q_values.argmax().item()
    
    def get_action(self, driver, race_state, upcoming_zone=None) -> DriverAction:
        """Get action for the driver at a decision point.
        
        Args:
            driver: DriverState object
            race_state: RaceState object
            upcoming_zone: Optional upcoming overtaking zone info
            
        Returns:
            DriverAction with risk level and overtake decision
        """
        # Create feedback from current state
        feedback = create_driver_feedback(driver, race_state, upcoming_zone)
        state = feedback.to_vector(normalize=True)
        
        # Select action
        # by having explore == self.training_mode, in eval mode we are now choosing greedy actions
        action_idx = self.select_action(state, explore=self.training_mode)
        
        # Store for learning (will be used when we get reward)
        self.last_state = state
        self.last_action_idx = action_idx
        self.last_feedback = feedback
        
        # Decode action
        risk, attempt = self._decode_action(action_idx)
        return DriverAction(risk_level=risk, attempt_overtake=attempt)
    
    def store_transition(self, reward: float, next_driver, next_race_state, done: bool, next_zone):
        """Store a transition in the replay buffer.
        
        Args:
            reward: Reward received for last action
            next_driver: DriverState after action
            next_race_state: RaceState after action
            done: Whether episode is finished
            next_zone: upcoming zone for next state
        """
        if self.last_state is None or self.last_action_idx is None:
            return  # No previous action to learn from
        
        # Get next state
        next_feedback = create_driver_feedback(next_driver, next_race_state, next_zone)
        next_state = next_feedback.to_vector(normalize=True)
        
        # Store transition
        self.replay_buffer.push(
            self.last_state,
            self.last_action_idx,
            reward,
            next_state,
            done
        )
    
    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """Perform one training step with a batch from replay buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # Compute current Q-values
        q_values = self.model(states).gather(1, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = nn.functional.mse_loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()
    
    def on_episode_end(self):
        """Call this at the end of each race episode to decay epsilon once per episode."""
        if self.training_mode:
            self.episodes_completed += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def set_training_mode(self, training: bool):
        """Set agent to training or evaluation mode."""
        self.training_mode = training
        if training:
            self.model.train()
        else:
            self.model.eval()
    
    def save(self, filepath: str):
        """Save model weights and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
        }, filepath)
    
    def load(self, filepath: str):
        """Load model weights and training state."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
    
    def _decode_action(self, action_idx: int):
        """Map action index to (RiskLevel, attempt_overtake)."""
        return self.ACTION_SPACE[action_idx]