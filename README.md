# 🐦 Orange Pigeon: Smart Pest Deterrence Matrix (OpenEnv Hackathon)

## Overview
This is a custom Reinforcement Learning (RL) environment built for the **Meta PyTorch OpenEnv Hackathon**. The "Orange Pigeon" environment simulates an automated pest deterrence system for smart cities, warehouses, or airports. 

An AI agent must learn to deter pigeons using different sound frequencies while actively minimizing noise pollution.

## The Environment (Gymnasium API)
- **State Space (MultiDiscrete):** Represents whether a pigeon is present `[0 or 1]` and the current noise level in the area `[0 to 10]`.
- **Action Space (Discrete 3):**
  - `0`: Do nothing (allows noise to decrease).
  - `1`: Play Low-Frequency Sound (50% chance to deter, slight noise increase).
  - `2`: Play High-Frequency Sound (90% chance to deter, high noise increase).

## Reward Logic
- **+10** for successfully deterring the pigeon.
- **-1** if the deterrent fails.
- **-5** penalty for causing noise pollution when no pigeon is present.
- **-5** penalty for exceeding the safe noise threshold (>7).

## How to Run
1. Install requirements: `pip install gymnasium numpy`
2. Test the environment with a random agent: `python test_agent.py`