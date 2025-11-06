# F1_StrategySimulator
This is a Final Year Project for my Bachelors in Computer Science in the University of Leeds.

## Aim 
I want to investigate the effectiveness of different MARL (Multi-agent reinforcement learning) on-policy or off-policy model-free algorithms in competitive stochastic environments such as in the context of F1 races. 
Race strategy in Formula One is a complex optimisation problem influenced by unpredictable race events, interactions with competitors, and evolving technical regulations, making it stochastic in nature. 
The investigation will range from the robustness of algorithms to converge on policies when introducing environment randomness, to comparing the speed at which different algorithms arrive at their policies when put against each other. 
The algorithms to begin investigation on are independent DQN or policy gradients per agent to verify convergence, then potentially expand to compare against a MADDPG-style actor-critic.

### Secondary Aim
While that's my main project aim, I personally want to explore F1 strategy when we don't discretise lap-by-lap and try and find out what algorithm works best in stochastic environments, this helps us replicate the true nature of F1 and motorsport in general.

## How to run
1. Install requirements from requirements.txt (preferably in a Virtual Environment such as Conda)
2. Run the cells in f1StrategySimulatorMVP Jupyter to check if the MVP works.

