# 🚖 Deep Q-Network (DQN) on Taxi-v3  

This repository contains a **from-scratch implementation of Deep Q-Network (DQN)** applied to the **Taxi-v3** environment from OpenAI Gym.  

The goal is to demonstrate how DQN learns to pick up and drop off passengers optimally in a discrete environment, by combining **Q-learning** with **deep neural networks** and **experience replay**.  

---

## 📌 About DQN  

- Introduced in *"Playing Atari with Deep Reinforcement Learning"* (Mnih et al., 2013).  
- Key innovations:  
  - **Experience Replay** → stores transitions `(s, a, r, s')` in a buffer and samples mini-batches for training.  
  - **Target Network** → stabilizes learning by using a separate Q-network for bootstrapping.  
- Objective: minimize the Temporal Difference (TD) error:  

---
## 📈 Results

- The DQN agent learns to minimize penalties and efficiently complete the task.

- Training logs & metrics are tracked using Weights & Biases (wandb).

[🔗WandB Logs](https://wandb.ai/atharv3105-dr-a-p-j-abdul-kalam-technical-university/DQN_Taxi/runs/e5xnattq)

---
## References

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
