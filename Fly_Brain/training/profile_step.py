import time
from game.flappy import FlappyBirdEnv
from game.agent import FlyBrainAgent
import torch

env = FlappyBirdEnv(render=False)
agent = FlyBrainAgent()

obs = env.reset()
agent.reset()

start = time.time()
log_probs, rewards = [], []
for step in range(300):
    action, log_prob = agent.act(obs)
    obs, reward, done, info = env.step(action)
    log_probs.append(log_prob)
    rewards.append(reward)
    if done:
        break
forward_time = time.time() - start

start = time.time()
returns = torch.tensor(rewards, dtype=torch.float32)
loss = -(torch.cat(log_probs) * returns).sum()
loss.backward()
backward_time = time.time() - start

print(f"Forward (rollout) time: {forward_time:.2f}s over {step+1} steps")
print(f"Backward (loss.backward()) time: {backward_time:.2f}s")