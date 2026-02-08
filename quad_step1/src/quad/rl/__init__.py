"""
Reinforcement-learning tooling for the quadrotor Gymnasium environment.

Requires the optional ``rl`` extras::

    uv sync --extra rl
    # or
    pip install -e ".[rl]"

Submodules
----------
config      – Dataclass configs + argparse loaders
wrappers    – Lightweight Gym wrappers (clip, normalise)
baselines   – Zero / random policy evaluation
train_ppo   – Stable-Baselines3 PPO training loop
eval_sb3    – Evaluate a saved SB3 model
"""
