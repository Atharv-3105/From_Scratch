# Generated from: train_ppo_viz_doom_kaggle.ipynb
# Converted at: 2026-07-28T12:13:33.631Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ## PPO Understanding


# ### Before PPO?
# - While using classic Policy Gradient Algorithm training was unstable as it's objective was *"If an action produced high-reward, increase its probability"* as model follwed high-reward it can abruptly change its policy which lead to collapse in training, massive variance.


# #### TRPO(Trust Region Policy Optimization)
# - It introduced *"Don't allow the policy to move too far in one update"* meaning a threshold was set that policy can change upto this only, but implementing this was very complex.


# ### PPO(Proximal Policy Optimization)
# - It introduced *"Let Gradient Descent improve the policy, but automatically ignore updates that try to change the policy too much."*
# - PPO is an **on-policy algorithm** because it uses data collected by the current(or very recent) policy, and it prevents that policy from drifting too far while reusing the same batch.


# ## PPO Implementation


!pip install --upgrade vizdoom gymnasium wandb imageio opencv-python

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np 
import gymnasium as gym 
from gymnasium import spaces 
import wandb
import cv2 
import os 
from vizdoom import gymnasium_wrapper 
import imageio

class Config:
    def __init__(self):
        self.env_name = "VizdoomDefendCenter-v1"
        self.total_timesteps = 500000
        self.learning_rate = 2.5e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.epochs = 4
        self.batch_size = 128
        self.buffer_size = 2048
        self.hidden_size = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_stack = 4
        self.video_log_interval = 50
        self.num_actions = 3
        
config = Config()
wandb.init(project = "ppo-vizdoom", config = vars(config), mode = 'online')
        

class WandbVideoRecorder(gym.Wrapper):
    """ 
        This wrapper will grab the RBG frames at every step and at the end of an episode
        will stitch those frames into an .mp4
    """
    def __init__(self,env, interval = 50):
        super().__init__(env)   
        self.interval = interval 
        self.episode_count = 0
        self.recording = False 
        self.frames = []

    def _get_render_frame(self):
        """ 
            Helper to safely extract the RGB array from render()
        """
        frame = self.env.render()

        if isinstance(frame, dict):
            frame = frame.get('rgb', frame.get('screen', None))

        #Ensure it's a proper numpy array
        if frame is not None:
            frame = np.array(frame, dtype = np.uint8)
            
            # ViZDoom channels transpose check: (C, H, W) -> (H, W, C)
            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                frame = np.transpose(frame, (1, 2, 0))
        return frame
    
    def reset(self, **kwargs):
        if self.episode_count % self.interval == 0:
            self.recording = True 
            self.frames = []
        else:
            self.recording = False 
            
        obs, info = self.env.reset(**kwargs)
        
        #If recording, grab the first frame
        if self.recording:
            frame = self._get_render_frame()
            if frame is not None:
                self.frames.append(frame)
                
        return obs, info 
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.recording:
            frame = self._get_render_frame()
            if frame is not None:
                self.frames.append(frame)

        done = terminated or truncated
        if done:
            if self.recording:
                self.save_and_log_video()
            else:
                self.episode_count += 1
                
        return obs, reward, terminated, truncated, info 
    
    def save_and_log_video(self):
        if self.recording and len(self.frames) > 0:
            video_path = f"vizdoom_ep_{self.episode_count}.mp4"
            imageio.mimsave(video_path, self.frames, fps = 30)
        
            wandb.log({
                "gameplay_video": wandb.Video(video_path, fps = 30, format = "mp4"),
                "episode": self.episode_count
            })
            print(f"-------Successfully logged videos for Episode {self.episode_count}----")
        self.episode_count += 1
        self.recording = False 
        self.frames = []
        

class ImagePreprocessingWrapper(gym.Wrapper):
    """ 
        A wrapper to perform image pre-processing operations 
    """
    def __init__(self, env, frame_stack = 4):
        super().__init__(env)
        self.frame_stack = frame_stack
        #Observation Shape
        self.obs_shape = (84, 84)
        
        #Override the Observation_Space inorder to match stacked grayscale frames
        self.observation_space = spaces.Box(low = 0, high = 1.0, shape = (frame_stack, 84, 84), dtype = np.float32)
        self.frames = []
        
    def _preprocess(self, obs):
        if isinstance(obs, dict):
            obs = obs.get("screen", obs.get("rgb", None)) # Extract the image array, ignore the rest
            if obs is None:
                raise KeyError("Observation dict missing both 'screen' and 'rgb' keys")
            
        # 2. If for some reason it's still a tuple/list, grab the first element
        if isinstance(obs, (tuple, list)):
            obs = obs[0]

        
        #Observation comes in as (240, 320, 3) numpy array i.e. (width, height, RGB)
        obs = np.array(obs, dtype = np.uint8)

        # 3. Transpose (C, H, W) -> (H, W, C) if needed
        if obs.ndim == 3 and obs.shape[0] in (1, 3, 4):
            obs = np.transpose(obs, (1, 2, 0))
            
        # 4. Convert to Grayscale
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 3 and obs.shape[2] == 1:
            gray = obs.squeeze(-1)
        else:
            gray = obs
        
        #RESIZE TO 84 x 84
        resized = cv2.resize(gray, self.obs_shape, interpolation = cv2.INTER_AREA)
        
        #NORMALIZE to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._preprocess(obs)
        
        #Fill frame stack with the first frame
        self.frames = [processed for _ in range(self.frame_stack)]
        return np.array(self.frames, dtype = np.float32), info 
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._preprocess(obs)
        
        #Append new frames, remove the oldest
        self.frames.append(processed)
        self.frames.pop(0)
        
        return np.array(self.frames, dtype = np.float32), reward, terminated, truncated, info
    

class RewardShaper(gym.Wrapper):
    def __init__(self, env, fire_action_id = 2, ammo_key = "ammo"):
        super().__init__(env)
        self.fire_action_id = fire_action_id
        self.ammo_key = ammo_key
        self.previous_ammo = 50

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if isinstance(info, dict) and self.ammo_key in info:
            self.previous_ammo = info[self.ammo_key]
        else:
            self.previous_ammo = 50 
    
        return obs, info 

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = original_reward

        #Penalty for shooting (prevents spamming shots blindly)
        if action == self.fire_action_id:
            shaped_reward -= 0.01

        #Reward for successfull kills
        if original_reward > 0:
            shaped_reward += 0.5 #Modest bonus

        #Ammo-delta tracking(if info contains ammo state)
        if isinstance(info, dict) and self.ammo_key in info:
            current_ammo = info[self.ammo_key]
            ammo_used = self.previous_ammo - current_ammo

            #if ammo was spent without getting a kill,apply a small extra penalty
            if ammo_used > 0 and original_reward <= 0:
                shaped_reward -= 0.05 * ammo_used

            self.previous_ammo = current_ammo

        return obs, shaped_reward, terminated, truncated, info

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        
        #------------CNN Based Feature Extractor-------------
        #Input: (4, 84, 84) -> Output: (64, 7, 7) --> Flatten --> 3136
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, config.hidden_size),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(config.hidden_size, num_actions)
        # self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Linear(config.hidden_size, 1)
        
    
    def forward(self, x):
        features = self.shared(x) #Shape: [Batch, 4, 84, 84]
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, obs):
        #Obs comes in as an Numpy Array
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(config.device)  #We also add batch_dim SHAPE:[1, 4, 84, 84]
        logits, value = self.forward(obs_tensor)
        #Use categorical distribution instead of Normal
        dist = torch.distributions.Categorical(logits = logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        action = action.squeeze(0) #Remove batch dim from env
        return action.cpu().detach().numpy(), log_prob.cpu().detach().item(), value.cpu().detach().item()
    
    def evaluate(self, obs, action):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits = logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy    
        

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def get(self):
        data = {
            "obs": torch.FloatTensor(np.array(self.obs)).to(config.device),
            "actions": torch.LongTensor(np.array(self.actions)).to(config.device),
            "rewards": torch.FloatTensor(np.array(self.rewards)).to(config.device),
            "dones": torch.FloatTensor(np.array(self.dones)).to(config.device),
            "log_probs": torch.FloatTensor(np.array(self.log_probs)).to(config.device),
            "values": torch.FloatTensor(np.array(self.values)).to(config.device)
        }
        
        self.clear()
        return data 
    
    def clear(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.dones, self.log_probs, self.values = [], [], []

# #### GAE(Generalized Advantage Estimation)
# - Temporal Difference Error represents the immediate surprise in reward plus discounted future value.
# - GAE balances variance and bias by taking an exponentially weighted average of k-step advantages.


def compute_gae(buffer_data, last_value):
    #Get the trajectory data collected during the rollout
    rewards = buffer_data["rewards"]    #This is the immediate rewards r_t
    values = buffer_data["values"]      #This is the Critic's esitmated state values V(s_t)
    dones = buffer_data["dones"]        #This is the termination flags
    
    #Initialize the advantage tensor with zeros matching the shape of the Rewards tensor
    advantages = torch.zeros_like(rewards).to(config.device)
    last_gae = 0
    
    #Iterate backwards through time: t = T-1, T-2, T-3,.....,0
    for t in reversed(range(len(rewards))):
 
        #Determine the Value{s_{t+1}} for the next step
        if t == len(rewards) - 1:
            next_value = last_value     #Value of the state reached after the final step
        else:
            next_value = values[t + 1]
        
        #Calculate Temporal Difference(TD) error: delta_at_t = reward_at_t + (gamma * Value_at_step_t+1 *(1 - done_at_t)) - Value_at_step_t
        delta = rewards[t] + config.gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + config.gamma * config.gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
        
    returns = advantages + values 
    return advantages , returns  

def ppo_update(policy, optimizer, buffer_data, advantages, returns):
    obs = buffer_data["obs"]
    actions = buffer_data["actions"]
    old_log_probs = buffer_data["log_probs"]
    
    #Normalize Advantages to have mean 0 and standard deviation 1 to keep gradient updates consistent
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    dataset_size = len(obs)
    indices = np.arange(dataset_size)
    
    policy_losses, value_losses, entropies = [], [], []
    
    #Mini-Batch Optimization Loop
    for _ in range(config.epochs):
        np.random.shuffle(indices)
        
        #Slice the dataset into mini-batches
        for start in range(0, dataset_size, config.batch_size):
            end = start + config.batch_size
            batch_idx = indices[start:end]
            
            #Slice mini-batch data 
            b_obs = obs[batch_idx]
            b_actions = actions[batch_idx]
            b_old_log_probs = old_log_probs[batch_idx]
            b_advantages = advantages[batch_idx]
            b_returns = returns[batch_idx]
            
            log_prob, value, entropy = policy.evaluate(b_obs, b_actions)
            
            #Compute probability-ratio r_t(theta)
            ratio = torch.exp(log_prob - b_old_log_probs)
            
            #Unclipped objective element
            surr1 = ratio * b_advantages 
            
            #Clipped objective element
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * b_advantages
            
            #PPO Clipped Surrogate Loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            #Value function i.e Critic Loss using MSE
            value_loss = nn.MSELoss()(value, b_returns)
            
            #Mean Policy Entropy
            entropy_loss = entropy.mean()
            
            #Combined Loss Function in which model will try to minimize the policy,value loss while maximizing the entropy
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            #Gradient clipping
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_loss.item())
            
            
    return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)        

def train():
    
    env = gym.make(config.env_name, render_mode = "rgb_array")
    env = RewardShaper(env)
    env = WandbVideoRecorder(env, interval = config.video_log_interval)
    env = ImagePreprocessingWrapper(env, frame_stack = config.frame_stack)
    
    policy = ActorCritic(config.num_actions).to(config.device)
    optimizer = optim.Adam(policy.parameters(), lr = config.learning_rate)
    buffer = RolloutBuffer()
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    print(f"Starting VizDoom training for {config.total_timesteps} timesteps...")
    print(f"Using Device: {config.device}")
    
    for timestep in range(1, config.total_timesteps + 1):
        action, log_prob, value = policy.get_action(obs)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated 
        
        buffer.add(obs, action, reward, done, log_prob, value)
        
        obs = next_obs 
        episode_reward += reward 
        episode_length += 1 
        
        #Update PPO
        if timestep % config.buffer_size == 0:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(config.device)
                _, last_value = policy.forward(obs_tensor)
                last_value = last_value.cpu().item()
                
            buffer_data = buffer.get()
            advantages, returns = compute_gae(buffer_data, last_value)
            
            avg_pol_loss, avg_val_loss, avg_entropy = ppo_update(policy, optimizer, buffer_data, advantages, returns)
            
            wandb.log({
                "timesteps": timestep,
                "update/policy_loss": avg_pol_loss,
                "update_value_loss": avg_val_loss,
                "update_entropy": avg_entropy
            })
            
        if done:
            episode_count += 1
            wandb.log({
                "episode": episode_count,
                "episode_reward" : episode_reward,
                "episode_length": episode_length
            })
            
            if episode_count % 10 == 0:
                print(f"Timestep: {timestep} | Episode: {episode_count} | Reward: {episode_reward:.2f}")
                
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
    env.close()
    wandb.finish()
    print(f"VizDoom Training Completed")

train()