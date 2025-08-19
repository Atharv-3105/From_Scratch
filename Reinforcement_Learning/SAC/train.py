import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import gymnasium as gym 
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
import os
import random
from tqdm import tqdm
import time
import numpy as np
import wandb
import imageio


class Config:
    exp_name = "Pendulum-SAC" #
    seed = 42
    env_id = "Pendulum-v1" #  Gym Environment Name
    
   
    alpha = 0.2  #Entropy Co-efficient
    
    # Action space for Pendulum-v1 is [-2.0, 2.0]
    low = -2.0 
    high = 2.0 
    
    #Training Parameters
    total_timesteps = int(2e5) 
    learning_rate = 3e-4
    buffer_size = 100000
    gamma = 0.99
    tau = 0.005 #For soft_update of Q-Networks for stable_training
    target_network_freq = 1
    batch_size = 256
    learning_starts = int(5e3) # Reduced learning starts proportionally
    train_freq = 2
    
    #Logging
    capture_video = True
    save_model = True
    upload_model = True
    #WandB configuration
    use_wandb = True
    wandb_project = "cleanRL"
    wandb_username = ""
    
    

class ActorNet(nn.Module):
    '''
        This is the Actor Network
        It's job is to generate distribution over actions;
        from which the Policy will pick the stochastic actions for exploration.
    
    '''
    
    def __init__(self, state_space, action_space):
        super().__init__()
        print(f"State Space: {state_space}, Action_space:{action_space}")
        # For Pendulum, state_space is 3, action_space is 1.
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,16)
        self.mu = nn.Linear(16, action_space) #Map's the Hidden_Features to the mean of the Gaussian Action Distribution
        self.sigma = nn.Linear(16, action_space) #Map's the Hidden_Features to the Standard_Deviation
        
    def forward(self, x):
        #Pass the input x through 3 layers of Mish Activation;
        #Mish() is an Activation function like ReLU but better for continuous control.
        x = F.mish(self.fc1(x))
        x = F.mish(self.fc2(x))
        x = F.mish(self.fc3(x))
        
        #mu is the Mean of the Gaussian Action
        mu = self.mu(x)
        #passed through softplus() to make the Output positive{as std can't be negative}
        sigma = F.softplus(self.sigma(x))
        
        #This is a Normal Distribution
        return mu, sigma

    def get_action(self, x):
        '''
            This function will return:
                1-  Sampled Action
                2-  Log Probability(for actor loss)
                3-  Entroy
        '''
        
        mu, sigma = self.forward(x)
        dist = torch.distributions.Normal(mu, sigma)
        
        #Sample action using Reparameterization; As rsample() is differentiable;
        #Needed for backpropagation through Stochastic Nodes
        action = dist.rsample()
        
        # Bound actions to stay in range [-1,1] using tanh()
        # Then scale to the environment's specific action range (e.g., [-2, 2] for Pendulum)
        action_normalize = torch.tanh(action)
        
        # Adjust log_prob for the tanh squashing
        log_prob = dist.log_prob(action)
        log_prob = log_prob - torch.log(1 - action_normalize.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim = -1, keepdim = True)
        
        # Scale the normalized action to the environment's actual action space
        action_scaled = action_normalize * Config.high 
        
        entropy = dist.entropy()
        
        return action_scaled, log_prob, entropy # Return scaled action
        

class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        # Maps features from state_space to 256
        self.fc1 = nn.Linear(state_space, 256)
        
        # Maps features from action_space to 256
        self.fc2 = nn.Linear(action_space,256)
        
        # Concatenated features from state_space and action_space {256 + 256 = 512}
        self.fc3 = nn.Linear(512,512)
        self.reduce = nn.Linear(512, 256)
        # Map the features to a Scalar Q-Value
        self.out = nn.Linear(256,1)
    
    def forward(self, state, action):
        state = F.mish(self.fc1(state))
        action = F.mish(self.fc2(action))
        
        temp = torch.cat((state, action),dim=1)
        x = F.mish(self.fc3(temp))
        x = F.mish(self.reduce(x))
        x = self.out(x)
        return x
        

def make_env(env_id, seed, capture_video, run_name, eval_mode = False, render_mode = None):
    '''
        Create Environment with Video_Recording
    '''
    # Determine the render_mode based on capture_video and eval_mode
    current_render_mode = render_mode
    if capture_video and not eval_mode:
        # If recording training video, render_mode is rgb_array
        current_render_mode = "rgb_array"

    env = gym.make(env_id, render_mode= current_render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    #Video Recording Setup
    if capture_video:
        if eval_mode:
            video_prefix = f"video/{run_name}/eval"
        else:
            video_prefix = f"video/{run_name}/train"
            env = gym.wrappers.RecordVideo(
                env,
                video_prefix,
                # Record every 100 episodes for Pendulum, as episodes are shorter
                episode_trigger=lambda x: x % 100 ==0 
            )
    
    env.action_space.seed(seed)
    
    return env


def evaluate(model, device, run_name, num_eval_eps = 10, record = False):
    
    eval_env = make_env(Config.env_id, seed=Config.seed, capture_video=True, run_name=run_name, eval_mode=True, render_mode="rgb_array")
    eval_env.action_space.seed(Config.seed)
    
    model = model.to(device)
    #Set Model to evaluation
    model = model.eval()
    
    returns, frames = [],[]
    
    for eps in range(num_eval_eps):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            
            if(record):
                # 
                # Don't consider Early Stopping Condition as Pendulum rewards are typically negative and don't exceed 500.
                # if(episode_reward > 500):
                #     print("Episode Reward exceeded 500; Stopping Early")
                #     break
                frame = eval_env.render()
                frames.append(frame)
                
            with torch.no_grad():
                # In evaluation, SAC policy is typically deterministic (mean of the distribution)
                action, _, _ = model.get_action(torch.tensor(obs, device = device).unsqueeze(0))
                
                #clip action for robustness if action_scaled goes slightly out of bounds
                action = torch.clip(action, Config.low, Config.high) 
                
                #Convert action to numpy
                action_np = action.cpu().numpy().flatten()
            
            obs,reward,terminated,truncated,_ = eval_env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
        
        returns.append(episode_reward)
    eval_env.close()
    
    #Save video
    if frames:
        os.makedirs(f"videos/{run_name}/eval",exist_ok=True)
        imageio.mimsave(
            f"videos/{run_name}/eval/eval_video.mp4",
            frames,
            fps = 30
        )
    
    return returns,frames

def main():
    
    config = Config()
    run_name = f"{config.exp_name}__{int(time.time())}"
    
    #Initialize WandB
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_username,
            sync_tensorboard=True,
            config = vars(config),
            name = run_name,
            monitor_gym=True,
            save_code=True)
    
    os.makedirs(f"videos/{run_name}/train",exist_ok=True)
    os.makedirs(f"videos/{run_name}/eval",exist_ok=True)
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")
    
    #Set THE SEEDS
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    env = make_env(config.env_id, seed=config.seed, capture_video=config.capture_video, run_name=run_name,)
    
    actor_net = ActorNet(state_space= env.observation_space.shape[0], action_space=env.action_space.shape[0]).to(device)
    
    q1_network = QNet(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0]).to(device)
    q2_network = QNet(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0]).to(device)
    
    #Target Q-Networks which will be used for Q-functions
    target_q1_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    target_q2_network = QNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    
    target_q1_network.load_state_dict(q1_network.state_dict())
    target_q2_network.load_state_dict(q2_network.state_dict())
    
    actor_optim = optim.Adam(actor_net.parameters(), lr=config.learning_rate)
    q1_optim = optim.Adam(q1_network.parameters(), lr=config.learning_rate)
    q2_optim = optim.Adam(q2_network.parameters(), lr=config.learning_rate)
    
    actor_net.train()
    q1_network.train()
    q2_network.train()
    
    replay_buffer = ReplayBuffer(buffer_size=config.buffer_size,
                                 observation_space=env.observation_space,
                                 action_space=env.action_space,
                                 device=device,
                                 handle_timeout_termination=False)
    
    obs, _ = env.reset()
    start_time = time.time()
    
    for step in tqdm(range(config.total_timesteps)):
        
        with torch.no_grad():
            # Get action from actor, returns scaled action
            action, _, entropy = actor_net.get_action(torch.tensor(obs, device=device).unsqueeze(0))
            
        # Action is already scaled by get_action, convert to numpy
        action_np = action.cpu().numpy().flatten()
        
        new_obs , reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        replay_buffer.add(obs, new_obs, action_np, np.array(reward), np.array(done), [info]) # Use action_np here
        
        if "episode" in info:
            print(f"Step:{step}, Return:{info['episode']['r']}")
            
            #WandB logging
            if config.use_wandb:
                wandb.log({
                    "episodic_return":info['episode']['r'],
                    "episodic_length":info['episode']['l'],
                    "action":action_np, # Log the actual action taken by the environment
                    "global_step":step
                })    
        if step > config.learning_starts:
            data = replay_buffer.sample(config.batch_size)
            
            
            with torch.no_grad():
                # next_actions are sampled from the current policy for target Q calculation
                next_actions, log_prob, entropy = actor_net.get_action(data.next_observations)
                
                target_max1 = target_q1_network(data.next_observations, next_actions)
                target_max2 = target_q2_network(data.next_observations, next_actions)
                target_max = torch.min(target_max1, target_max2)
                
                # Use Entropy coefficient in the soft target calculation
                soft_target = target_max - config.alpha * log_prob 
                td_target = data.rewards + config.gamma * soft_target * (1 - data.dones)
                
            q_val1 = q1_network(data.observations, data.actions)
            q_val2 = q2_network(data.observations, data.actions)
            q1_optim.zero_grad()
            loss1 = F.mse_loss(q_val1, td_target)
            loss2 = F.mse_loss(q_val2, td_target)
            
            loss1.backward(retain_graph=True)
            q1_optim.step()
            q2_optim.zero_grad()
            loss2.backward()
            q2_optim.step()
            
            if step % config.train_freq == 0:
                
                actions, log_probs, entropy = actor_net.get_action(data.observations)
                action_val1 = q1_network(data.observations, actions)
                action_val2 = q2_network(data.observations, actions)
                action_values = torch.min(action_val1, action_val2)
                
                # Actor loss: maximize Q-value minus entropy term
                loss = action_values - config.alpha * log_probs 
                loss = -loss.mean() # Minimize negative loss
                
                actor_optim.zero_grad()
                loss.backward()
                actor_optim.step()
            
            #Log loss and metrics at every 100 step     
            if step % 100 == 0:
                if config.use_wandb:
                    wandb.log({
                        "losses/td_loss1":loss1.item(),
                        "losses/td_loss2": loss2.item(),
                        "losses/actor_loss": loss.item(), # Log actor loss
                        "charts/SPS": int(step / (time.time() - start_time)), # Steps per second
                        "global_step": step
                    })
                
            #Update target network
            
            if step % config.target_network_freq == 0:
                for q_params, target_params in zip(q1_network.parameters(), target_q1_network.parameters()):
                    target_params.data.copy_(config.tau * q_params.data + (1.0 - config.tau) * target_params.data)
                
                for q_params, target_params in zip(q2_network.parameters(), target_q2_network.parameters()):
                    target_params.data.copy_(config.tau * q_params.data + (1.0 - config.tau) * target_params.data)
        
        if step % 500 == 0:
            
            episodic_retuns, eval_frames = evaluate(actor_net, device, run_name)
            avg_return = np.mean(episodic_retuns)
            
            if config.use_wandb:
                wandb.log({
                    "Val_Avg_Returns":avg_return,
                    "Val_Step":step
                })
            print(f"Evaluation Retuns:{episodic_retuns}")
        
        if done:
            obs, _ = env.reset()
        else:
            obs = new_obs
    env.close()
    writer.close()
    
    if config.use_wandb:
        final_video_path = f"videos/SAC_{config.env_id}.mp4"
        returns , frames = evaluate(actor_net, device, run_name, record=True)
        imageio.mimsave(
                final_video_path,
                frames,
                fps = 30,
                codec = 'libx264'
        )
        print(f"Final Video saved to:{final_video_path}")
        wandb.finish()
    
    
if __name__ == "__main__":
    main()
