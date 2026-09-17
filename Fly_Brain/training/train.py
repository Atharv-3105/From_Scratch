import torch 
import torch.optim as optim 
import numpy as np 

from game.flappy import FlappyBirdEnv
from game.agent import FlyBrainAgent

GAMMA = 0.99
LEARNING_RATE = 1e-3
NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 1000

def compute_discounted_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G 
        returns.insert(0, G)
        
    returns = torch.tensor(returns, dtype = torch.float32)
    #Baseline: center returns to reduce gradient variance
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns 



def train():
    env = FlappyBirdEnv(render = False)  
    agent = FlyBrainAgent()
    
    # Only the encoder + decoder are trainable -- the connectome buffers
    # inside agent.sim are registered as buffers, NOT parameters, so this
    # optimizer literally cannot touch them even by accident.
    optimizer = optim.Adam(list(agent.encoder.parameters()) + list(agent.decoder.parameters()),lr=LEARNING_RATE)
    
    episode_scores = []
    
    for episode in range(NUM_EPISODES):
        obs = env.reset()
        agent.reset()
        
        log_probs = []
        rewards = []
        
        for step in range(MAX_STEPS_PER_EPISODE):
            action, log_prob = agent.act(obs)
            obs, reward, done, info = env.step(action)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                break
            
        returns = compute_discounted_returns(rewards, GAMMA)
        log_probs_t = torch.cat(log_probs)
        
        loss = -(log_probs_t * returns).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_scores.append(info["score"])
        
        if episode % 10 == 0:
            recent_avg = np.mean(episode_scores[-10:])
            print(f"Episode {episode}: score={info['score']}, "
                  f"loss={loss.item():.3f}, avg_score(last 10)={recent_avg:.2f}")
            
    env.close()
    torch.save({"encoder": agent.encoder.state_dict(),
                "decoder": agent.decoder.state_dict()},
               "trained_agent.pt")
    print("Training complete. Saved to trained_agent.pt")
    
if __name__ == "__main__":
    train()