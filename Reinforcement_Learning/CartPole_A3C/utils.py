import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
# from huggingface_hub import HfApi
import matplotlib.pyplot as plt 
import os

#============================Function To Evaluate the Agent============================
def evaluate(global_model, env_name = "CartPole-v1", episodes = 5, render = False):
    env = gym.make(env_name)
    total_reward = 0
    
    for _ in range(episodes):
        state , _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits , _ = global_model(state_tensor)
                
            #We pick the action using Greedy Approach
            action = torch.argmax(logits, dim=-1).item()
            
            next_state, reward, terminated, truncated , _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            if render:
                env.render()
        total_reward += episode_reward
        
    env.close()
    return total_reward / episodes

# ============================Function To Record the Video of Our Agent============================
def record_video(global_model, env_name = "CartPole-v1", video_folder = "./videos", episodes = 1, repo_id =None):
    env = gym.make(env_name, render_mode = "rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    
    for ep in range(episodes):
        state , _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                logits , _ = global_model(state_tensor)
            action = torch.argmax(logits, dim=-1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            ep_reward += reward
            state = next_state
            
        print(f"[Record] Episode:{ep+1} | Reward: {ep_reward}")
        
    env.close()
    
#============================Function to Plot the results of our Agent============================  
def plot_results(scores, max_rewards, policy_losses, value_losses, entropies, save_path = None):
    
    episodes , avg_rewards = zip(*scores)
    _ , max_rewards = zip(*max_rewards)
    _ , policy_loss = zip(*policy_losses)
    _ , value_loss = zip(*value_losses)
    _ , entropy = zip(*entropies)
    
    plt.figure(figsize=(12,10))
    
    #Plot Episodes V/S Avg-Rewards
    plt.subplot(2,2,1)
    plt.plot(episodes, avg_rewards, label="Avg Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Rewards")
    plt.title("Average Reward over Episode")
    plt.grid()
    plt.legend()
    
    #Plot Max-Reward V/S Episodes
    plt.subplot(2,2,2)
    plt.plot(episodes, max_rewards, label="Max Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Max Rewards")
    plt.title("Max Reward over Episode")
    plt.grid()
    plt.legend()
    
    #Plot Policy_Loss and Value Loss over Episodes
    plt.subplot(2,2,3)
    plt.plot(episodes, policy_loss, label="Policy Loss")
    plt.plot(episodes, value_loss, label="Value Loss")
    plt.title("Loss over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    
    #Plot Entropy over Episodes
    plt.subplot(2,2,4)
    plt.plot(episodes, entropy, label="Entropy", color="green")
    plt.title("Entropy (Exploration)")
    plt.xlabel("Episodes")
    plt.ylabel("Entropy")
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[Plots] saved successfully")
    
    plt.show()
    
            
            
    
        