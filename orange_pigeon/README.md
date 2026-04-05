---
title: Orange Pigeon
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

#  Orange Pigeon: Autonomous Avian Deterrence System

##  The Real-World Problem (Why this matters?)
Orange Pigeon is a digital twin of an **IoT-based agricultural and urban defense system**. Pigeons and avian pests cause millions of dollars in damages globally through:
* **Aviation:** Bird strikes at airport runways.
* **Agriculture:** Crop destruction in large-scale farming.
* **Urban Logistics:** Contamination in warehouse inventories.

This environment trains an RL Agent to act as an automated sentinel—detecting threats and managing deterrence dynamically without human intervention.

---

##  Reinforcement Learning Design

### 1. State Space (Observation)
The environment provides a continuous sensory input representing the physical world:
* `[pigeon_proximity, system_battery, active_noise_level]` 
* *Why?* The agent must balance the urgency of the threat (proximity) against resource management (battery and noise pollution).

### 2. Action Space
A discrete action space representing physical hardware triggers:
* `0`: **Standby** (Conserve battery, monitor).
* `1`: **Low-Frequency Ping** (Mild deterrence, low power).
* `2`: **High-Frequency Siren** (Maximum deterrence, high power drain).

### 3. The Reward Function (Adaptive Logic)
The environment uses a shaped reward system to encourage optimal behavior:
* **+1.0** for successfully neutralizing a close-proximity threat.
* **-0.5** (Penalty) for false alarms (noise pollution without a threat).
* **-0.2** (Penalty) for continuous battery drain.
* *Goal:* Maximize deterrence while minimizing environmental noise and power consumption.

---

##  Baseline Performance & Tasks

To prove learning capability, the environment challenges the agent with 3 increasing difficulty levels:
1.  **Task 1 (Easy): Basic Detection** - Trigger any response when proximity crosses the threshold.
2.  **Task 2 (Medium): Battery Conservation** - Deter the threat without dropping battery below critical levels.
3.  **Task 3 (Hard): Adaptive Escalation** - Start with low noise, escalate to high siren only if the pigeon approaches closer.

### Agent Comparison (Proof of Learning)
| Agent Type | Avg. Normalized Score | Behavior Pattern |
| :--- | :--- | :--- |
| **Random Baseline** | 0.15 - 0.25 | Spamming siren, high battery drain, false alarms. |
| **Qwen2.5-72B-Instruct** | **> 0.65** | Strategic escalation, conserves battery when idle. |

---

##  Setup and Automated Grading
Built strictly complying with the Meta OpenEnv standard.
1. Clone the repository: `git clone https://github.com/aviralsachdeva9-stack/orangepigeon`
2. Ensure API credentials are set: `export HF_TOKEN="your_token"`
3. Run the inference baseline: `python inference.py`
4. Run validation: `openenv validate`

*This environment is fully containerized and compatible with automated LLM graders via Hugging Face Spaces.*