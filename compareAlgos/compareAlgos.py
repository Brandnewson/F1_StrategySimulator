
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import DoubleDQN, QRDQN, DuelingDQN, PDPPO
from sb3_contrib import MaskablePPO
from sb3_contrib import DiscreteSAC


ALGORITHMS = {
    "DQN": DQN,
    "DoubleDQN": DoubleDQN,
    "DuelingDQN": DuelingDQN,
    "PPO": PPO,
    "A2C": A2C,
    "PDPPO": PDPPO,
}

# Add DiscreteSAC if available
if DiscreteSAC is not None:
    ALGORITHMS["DiscreteSAC"] = DiscreteSAC

TIMESTEPS = 100000
EVAL_EPISODES = 5

def train_and_log_rewards(algo_name, algo_class, env_id="LunarLander-v3", timesteps=TIMESTEPS):
    # All algorithms use the same discrete environment for fairness
    env = gym.make(env_id)
    model = algo_class("MlpPolicy", env, verbose=0)
    rewards = []
    if algo_name in ["PPO", "PDPPO"]:
        # For PPO, train in one batch and log rewards after training
        # PPO (and similar on-policy methods) must be trained in batches for stability and correctness.
        model.learn(total_timesteps=timesteps)
        obs, info = env.reset()
        for _ in range(timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            if done or truncated:
                obs, info = env.reset()
    else:
        # Off-policy: DQN, DoubleDQN, DuelingDQN, DiscreteSAC (step-by-step)
        obs, info = env.reset()
        for _ in range(timesteps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            model.learn(total_timesteps=1, reset_num_timesteps=False)
            if done or truncated:
                obs, info = env.reset()
    env.close()
    return model, np.array(rewards)

def evaluate_agent(model, env_id="LunarLander-v3", episodes=EVAL_EPISODES, algo_name="DQN"):
    env = gym.make(env_id)
    eval_rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # env.render()  # Uncomment to visualize
            if done or truncated:
                break
        eval_rewards.append(total_reward)
    env.close()
    return np.array(eval_rewards)

results = {}
for algo_name, algo_class in ALGORITHMS.items():
    print(f"Training {algo_name}...")
    model, train_rewards = train_and_log_rewards(algo_name, algo_class)
    eval_rewards = evaluate_agent(model, algo_name=algo_name)
    results[algo_name] = {
        "train_rewards": train_rewards,
        "eval_rewards": eval_rewards,
    }
    print(f"{algo_name} evaluation rewards: {eval_rewards}")

# Plot training reward curves
plt.figure(figsize=(12, 6))
for algo_name, data in results.items():
    # Smooth training rewards for visualization
    train_rewards = data["train_rewards"]
    if len(train_rewards) > 100:
        smoothed = np.convolve(train_rewards, np.ones(100)/100, mode='valid')
    else:
        smoothed = train_rewards
    plt.plot(smoothed, label=f"{algo_name} (train)")
plt.title("Training Reward Curves")
plt.xlabel("Timestep")
plt.ylabel("Reward (smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot evaluation episode rewards
plt.figure(figsize=(8, 5))
for algo_name, data in results.items():
    plt.bar(algo_name, np.mean(data["eval_rewards"]), yerr=np.std(data["eval_rewards"]), capsize=8)
plt.title("Mean Evaluation Reward (5 episodes)")
plt.ylabel("Mean Reward ± Std")
plt.tight_layout()
plt.show()

# Print convergence info
for algo_name, data in results.items():
    train_rewards = data["train_rewards"]
    # Speed to convergence: first timestep where smoothed reward > 200 (solved)
    if len(train_rewards) > 100:
        smoothed = np.convolve(train_rewards, np.ones(100)/100, mode='valid')
    else:
        smoothed = train_rewards
    solved_idx = np.argmax(smoothed > 200) if np.any(smoothed > 200) else -1
    if solved_idx != -1:
        print(f"{algo_name} solved at timestep {solved_idx}")
    else:
        print(f"{algo_name} did not solve within {TIMESTEPS} timesteps")