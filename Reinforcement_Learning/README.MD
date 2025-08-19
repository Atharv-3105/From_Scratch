# 🧠 Reinforcement Learning From Scratch  

This repository contains **from-scratch implementations of popular Reinforcement Learning (RL) algorithms**, coded in pure PyTorch/Numpy (no high-level RL libraries).  
The goal is to deeply understand the inner workings of RL by re-implementing algorithms and testing them on classic environments.  

---

## 📌 Implemented Algorithms  

- **[DQN](./DQN-Taxi/)**  
  - Environment: Taxi-v3  
  - Vanilla Deep Q-Learning applied to a discrete action space problem.  

- **[Dueling DQN](./Flappy_Bird_DQN/)**  
  - Environment: Flappy Bird  
  - Implements the dueling architecture to separate value and advantage estimation.  

- **[A3C (Asynchronous Advantage Actor-Critic)](./CartPole_A3C/)**  
  - Environment: CartPole-v1  
  - Multi-threaded actor-learners update a shared global model asynchronously.  

- **[Soft Actor-Critic (SAC)](./SAC/)**  
  - Environment: Pendulum
  - Off-policy maximum-entropy RL for stable and exploratory learning.  

---


