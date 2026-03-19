import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from base_agents import BaseAgent, DriverAction, RiskLevel
from feedback import create_driver_feedback


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer used by rainbow-lite mode."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        max_priority = float(self.priorities.max()) if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = max(max_priority, 1e-6)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        valid_priorities = self.priorities[: len(self.buffer)]
        scaled = np.power(np.maximum(valid_priorities, 1e-6), self.alpha)
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.buffer), int(batch_size), p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = np.power(len(self.buffer) * probs[indices], -float(beta))
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx)] = float(max(priority, 1e-6))

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
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    """Dueling architecture with shared trunk and separate value/advantage heads."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.advantage_head = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        shared = self.net(x)
        value = self.value_head(shared)
        advantage = self.advantage_head(shared)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DQNAgent(BaseAgent):
    """DQN agent with config-driven variants for benchmarking."""

    ACTION_SPACE = [
        (RiskLevel.CONSERVATIVE, False),
        (RiskLevel.CONSERVATIVE, True),
        (RiskLevel.NORMAL, False),
        (RiskLevel.NORMAL, True),
        (RiskLevel.AGGRESSIVE, False),
        (RiskLevel.AGGRESSIVE, True),
    ]

    SUPPORTED_ALGOS = {"vanilla", "double", "dueling", "rainbow_lite"}

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
        algo: str = "vanilla",
        algo_options: Optional[Dict] = None,
    ):
        super().__init__(name=name)

        self.action_dim = len(self.ACTION_SPACE)
        self.state_dim = state_dim
        self.algo = str(algo or "vanilla").strip().lower()
        if self.algo not in self.SUPPORTED_ALGOS:
            raise ValueError(
                f"Unsupported dqn_params.algo '{self.algo}'. "
                f"Supported: {sorted(self.SUPPORTED_ALGOS)}"
            )
        self.algo_options = dict(algo_options or {})

        self.use_double_dqn = self.algo in {"double", "rainbow_lite"}
        self.use_dueling = self.algo in {"dueling", "rainbow_lite"}
        self.use_per = self.algo == "rainbow_lite"

        default_n_step = 3 if self.algo == "rainbow_lite" else 1
        self.n_step = int(self.algo_options.get("n_step", default_n_step))
        self.n_step = max(1, self.n_step)
        self.use_n_step = self.algo == "rainbow_lite" and self.n_step > 1
        self.n_step_buffer = deque(maxlen=self.n_step)

        per_alpha = float(self.algo_options.get("per_alpha", 0.6))
        self.per_beta_start = float(self.algo_options.get("per_beta_start", 0.4))
        self.per_beta_frames = int(self.algo_options.get("per_beta_frames", 100000))
        self.per_beta_frames = max(1, self.per_beta_frames)

        # Q-network and target network
        model_cls = DuelingDQN if self.use_dueling else SimpleDQN
        self.model = model_cls(state_dim, self.action_dim, hidden_size)
        self.target_model = model_cls(state_dim, self.action_dim, hidden_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Training components
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha)
        else:
            self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Hyperparameters
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.target_update_freq = int(target_update_freq)

        # Training state
        self.training_steps = 0
        self.episodes_completed = 0
        simulator_cfg = config.get("simulator", {}) if isinstance(config, dict) else {}
        self.training_mode = simulator_cfg.get("agent_mode") == "training"

        # Store last state/action for transition capture
        self.last_state = None
        self.last_action_idx = None
        self.last_feedback = None

    def _make_n_step_transition(self, transitions):
        reward_sum = 0.0
        done_flag = False
        last_next_state = transitions[-1][3]
        for idx, (_, _, reward, next_state, done) in enumerate(transitions):
            reward_sum += (self.gamma ** idx) * float(reward)
            last_next_state = next_state
            if done:
                done_flag = True
                break
        state, action = transitions[0][0], transitions[0][1]
        return state, action, reward_sum, last_next_state, done_flag

    def _push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _store_with_n_step(self, state, action, reward, next_state, done):
        if not self.use_n_step:
            self._push_transition(state, action, reward, next_state, done)
            return

        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        transition = self._make_n_step_transition(list(self.n_step_buffer)[: self.n_step])
        self._push_transition(*transition)

        if done:
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                transition = self._make_n_step_transition(list(self.n_step_buffer))
                self._push_transition(*transition)
            self.n_step_buffer.clear()
        else:
            self.n_step_buffer.popleft()

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self.training_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return q_values.argmax().item()

    def get_action(self, driver, race_state, upcoming_zone=None) -> DriverAction:
        feedback = create_driver_feedback(driver, race_state, upcoming_zone)
        state = feedback.to_vector(normalize=True)
        action_idx = self.select_action(state, explore=self.training_mode)

        self.last_state = state
        self.last_action_idx = action_idx
        self.last_feedback = feedback

        risk, attempt = self._decode_action(action_idx)
        return DriverAction(risk_level=risk, attempt_overtake=attempt)

    def store_transition(self, reward: float, next_driver, next_race_state, done: bool, next_zone):
        if self.last_state is None or self.last_action_idx is None:
            return

        next_feedback = create_driver_feedback(next_driver, next_race_state, next_zone)
        next_state = next_feedback.to_vector(normalize=True)

        self._store_with_n_step(
            self.last_state,
            self.last_action_idx,
            reward,
            next_state,
            done,
        )

    def _get_beta(self) -> float:
        if not self.use_per:
            return 1.0
        progress = min(1.0, self.training_steps / float(self.per_beta_frames))
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        if len(self.replay_buffer) < batch_size:
            return None

        if self.use_per:
            beta = self._get_beta()
            batch, indices, weights = self.replay_buffer.sample(batch_size, beta=beta)
            weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        else:
            batch = self.replay_buffer.sample(batch_size)
            indices = None
            weights_t = torch.ones((batch_size, 1), dtype=torch.float32)

        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.model(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.model(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_model(next_states_t).gather(1, next_actions)
            else:
                next_q = self.target_model(next_states_t).max(1, keepdim=True)[0]

            discount = self.gamma ** self.n_step if self.use_n_step else self.gamma
            target = rewards_t + discount * next_q * (1 - dones_t)

        td_error = target - q_values
        loss = (weights_t * (td_error ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_per and indices is not None:
            priorities = td_error.detach().abs().squeeze(1).cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)

        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return float(loss.item())

    def on_episode_end(self):
        if self.training_mode:
            self.episodes_completed += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_training_mode(self, training: bool):
        self.training_mode = training
        if training:
            self.model.train()
        else:
            self.model.eval()

    def save(self, filepath: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
                "algo": self.algo,
                "algo_options": self.algo_options,
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.training_steps = int(checkpoint.get("training_steps", self.training_steps))

    def _decode_action(self, action_idx: int):
        return self.ACTION_SPACE[action_idx]
