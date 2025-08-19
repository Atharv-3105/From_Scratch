import os
import random
import time 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
from huggingface_hub import HfApi, upload_folder
import cv2
from tqdm import tqdm
import wandb
import gymnasium as gym 
import imageio

from config import Config


args = Config()
#============Q-Network==============
class QNet(nn.Module):
    
    def __init__(self, state_space, action_space):
        super().__init__()
        print(f"State Space:{state_space}, Action_Space:{action_space}")
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256,512)
        self.q_value = nn.Linear(512, action_space)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        output = self.q_value(x)
        return output
        
#============Epsilon-Decay==============
class EpislonDecay(nn.Module):
    def __init__(self, initial_eps, end_eps, total_timesteps):
        super().__init__()
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.total_timesteps = total_timesteps
        
    def forward(self, current_timestep, decay_factor):
        slope = (self.end_eps - self.initial_eps)/(self.total_timesteps * decay_factor)
        return max(slope * current_timestep + self.initial_eps , self.end_eps)
    

#================Function to Create ENV=============
def make_env(env_name, seed, capture_video, run_name, eval_mode = False, render_mode = None):
    env = gym.make(env_name, render_mode = render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env.action_space.seed(seed)
    
    return env

#================Function to ONE-HOT ENCODE Observations=================
def one_hot_encode(obs):
    encoded = np.zeros(500, dtype = np.float32)
    encoded[obs] = 1.0
    return encoded


#======================Evaluate Function===================
def evaluate(model, device, run_name, num_eval_eps = 10, record = False, render_mode = None):
    eval_env = make_env(args.env_name, seed=args.seed, capture_video=True, render_mode=render_mode, run_name = run_name, eval_mode=True)
    eval_env.action_space.seed(args.seed)
    
    model = model.to(device)
    model = model.eval()
    
    returns = []
    frames = []
    
    for eps in tqdm(range(num_eval_eps)):
        obs , _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            
            if(record):
                if(episode_reward > 500):
                    print(f"Episode Reward Exceeded 500, Stopping Early.")
                    break
                frame = eval_env.render()
                frames.append(frame)
            
            obs = one_hot_encode(obs) #Convert State_space_size{i.e 16} to one-hot encoded vectors
            action = model(torch.tensor(obs, device=device).unsqueeze(0)).argmax().item()
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            action = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        returns.append(episode_reward)
    eval_env.close()
    
    #Save video
    if frames:
        os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
        imageio.mimsave(
            f"videos/{run_name}/eval/eval_video.mp4",
            frames,
            fps = 30
        )
    
    return returns, frames

run_name = f"{args.env_name}_{args.exp_name}_{args.seed}_{int(time.time())}"

#Inititalize WandB
if args.use_wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config = vars(args),
        name = run_name,
        monitor_gym=True,
        save_code = True
    )

os.makedirs(f"videos/{run_name}/train", exist_ok=True)
os.makedirs(f"videos/{run_name}/eval", exist_ok=True)
os.makedirs(f"runs/{run_name}", exist_ok=True)
writer = SummaryWriter(f"runs/{run_name}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = make_env(args.env_name, args.seed, args.capture_video, run_name)
state_space = env.observation_space.n 
action_space = env.action_space.n 
q_net = QNet(state_space, action_space).to(device)
target_net = QNet(state_space, action_space).to(device)
optimizer = optim.Adam(q_net.parameters(), lr=args.learning_rate)
eps_decay = EpislonDecay(args.start_e, args.end_e, args.total_timesteps)

q_net.train()
target_net.train()

replay_buffer = ReplayBuffer(args.buffer_size, gym.spaces.Box(low = 0, high= 1, shape=(500,), dtype=np.float32), env.action_space, device=device, handle_timeout_termination=False)

obs, _ = env.reset()
start_time = time.time()


for step in tqdm(range(args.total_timesteps)):
    
    eps = eps_decay(step, args.exploration_fraction)
    random_num = random.random()
    obs = one_hot_encode(obs) #16 is state_space size
    
    if random_num < eps:
        #Take a random_sample
        action = env.action_space.sample()
    else:
        action = q_net(torch.tensor(obs, device=device).unsqueeze(0)).argmax().item()
    
    new_obs, reward, terminated, truncated, info = env.step(action)
    
    #ONE_HOT_ENCODE new_observation
    new_obs_encoded = one_hot_encode(new_obs)
    
    done = terminated or truncated
    replay_buffer.add(
        obs,
        new_obs_encoded,
        np.array(action),
        np.array(reward),
        np.array(done),
        [info]
    )
    
    #TODO
    if "episode" in info:
        print(f"Step:{step}, Return:{info['episode']['r']}")
        
        #WandB logging
        if args.use_wandb:
            wandb.log({
                "episodic_return":info['episode']['r'],
                "episodic_length":info['episode']['l'],
                "epsilon":eps,
                "global_step":step
            })
    
    if step > args.learning_starts and step % args.train_frequency == 0:
        data = replay_buffer.sample(args.batch_size)
        
        target_max = target_net(data.next_observations).max(1)[0]
        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
        old_val = q_net(data.observations).gather(1, data.actions).squeeze()
        
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(old_val, td_target)
        
        loss.backward()
        optimizer.step()
        
        #Log LOSS & METRICS EVERY 100 steps
        if step % 100 == 0:
            if args.use_wandb:
                wandb.log({
                    "losses/td_loss":loss.item(),
                    "losses/q_values":old_val.mean().item(),
                    "step":step
                })
                
        if step % args.target_network_freq==0:
            for q_params,target_params in zip(q_net.parameters(), target_net.parameters()):
                target_params.data.copy_(args.tau * q_params.data + (1.0 - args.tau) * target_params.data)
            
    #==============MODEL EVALUATION & SAVING==============
    if args.save_model and step % 50000 == 0:
        
        #Evaluate Model
        episodic_returns , eval_frames = evaluate(q_net, device, run_name)
        avg_return = np.mean(episodic_returns)
        
        if args.use_wandb:
            wandb.log({
                "val_avg_return":avg_return,
                "val_step": step
            })
        print(f"Evaluation returns: {episodic_returns}")
    
    if done:
        obs, _ = env.reset()
    else:
        obs = new_obs
    
env.close()
writer.close()
    
    
if args.use_wandb:
    final_video_path = "videos/final.mp4"
    returns, frames = evaluate(q_net, device, run_name,record=True,num_eval_eps=4,render_mode="rgb_array")
    
    if os.path.exists(final_video_path) and os.listdir(final_video_path):
        wandb.log({"final_video":wandb.Video(f"{final_video_path}/rl-video-episode-0.mp4")})
    
    imageio.mimsave( final_video_path,frames, fps=30)
    print(f"Final Training video saved to:{final_video_path}")
    wandb.finish()

if args.capture_video:
    cv2.destroyAllWindows()
        

                