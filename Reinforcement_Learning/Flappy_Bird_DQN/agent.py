import warnings
warnings.filterwarnings("ignore")
import flappy_bird_gymnasium
import gymnasium as gym
from algo import DQN
from experience_replay import ReplayMemory
import torch
import torch.nn as nn

import itertools
import yaml
import os

from datetime import datetime, timedelta
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random

#For printing Date & Time
DATE_FORMAT = "%m-%d %H:%M:%S"

#Path to save the Agent Runs
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

#'Agg': Used to generate plots as images and save them to a file instead of logging to display
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameter_name):
        with open("hyperparameters.yml", 'r') as f:
            all_hyperparameter_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameter_sets[hyperparameter_name]
        
        self.hyperparameter_name = hyperparameter_name   
        self.replay_memory_size = hyperparameters['replay_memory_size'] #Size of the Replay_Memory Buffer
        self.mini_batch_size = hyperparameters['mini_batch_size']  #Size of the training data set from the replay_memory buffer
        self.epsilon_init = hyperparameters['epsilon_init']        #1 - 100% random actions
        self.epsilon_decay = hyperparameters['epsilon_decay']      #Epsilon Decay Rate
        self.epsilon_min = hyperparameters['epsilon_min']          #Minimum epsilon value
        self.network_sync_rate = hyperparameters['network_sync_rate']  #The minimum number of steps after which Policy Network will be synced with the Target Network
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.env_id = hyperparameters['env_id']
        
        #Define MeanSquaredError Loss Function & Optimizer Function
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        #Path to save Agents runs
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_name}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_name}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_name}.png')
    
    
    
    
    
    def run_agent(self, is_training = True, render = False):
        
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting......."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
        
        
        
        #Make Environment
        #Use **self.env_make_params to pass in environment-specific parameters
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)

        #Get the number of State in the Environment
        num_states = env.observation_space.shape[0]
        
        #Get the Number of Actions present in the Environment
        num_actions = env.action_space.n
        
        #Define 2 list which will store rewards per episode and epsilon history per episode
        rewards_per_episode = []
        epsilon_history = []
        
        #Initialize the DQN class
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        
        if is_training:
            #Initialize the Replay_Memory Buffer
            replay_memory = ReplayMemory(self.replay_memory_size)
            
            #Initialize the Epsilon Greedy Algorithm
            epsilon = self.epsilon_init
            
            #Initialize the Target_Q Network
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            
            
            #Track the number of steps taken: Used for syncing policy --> target network
            step_count = 0

            #Define the optimizer; Adam Optimizer will be perfect
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
            
            #Track best reward
            best_reward = -9999999
            
            
        else:
            #Load the learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            
            policy_dqn.eval()
            
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            
            terminated = False
            episode_reward = 0.0
            
            #Perform actions untill episode terminates or reaches max rewards
            while(not terminated and episode_reward < self.stop_on_reward):
                
                '''
                If when training in the starting the action will be random as epsilon will be at 100%,
                But with passing of episodes the Policy Network will learn the best policy for the maximum expected return on the reward,
                Hence when epsilon decreases the action will be picked by Policy DQ Network.
                '''
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    #Since we are not training; Hence we will disable the Gradients computation
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                
                #Processing
                new_state, reward, terminated, truncated, info = env.step(action.item())
                
                episode_reward += reward
                
                #Convert the new_state and rewards to tensor
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                if is_training:
                    #Save the experience into Replay Memory Buffer
                    replay_memory.append((state, action, new_state, reward, terminated))
                    
                    #Increment the step_counter
                    step_count += 1
                
                #Transition to next_state   
                state = new_state
            rewards_per_episode.append(episode_reward)
            
            #Save the model when new best_reward is obtained
            if is_training:
                
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New Best Reward {episode_reward:0.1f} ({(episode_reward - best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model....."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                    
                #Update Graph every second
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=15):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
                        

                #If enough experience has been collected
                if len(replay_memory) > self.mini_batch_size:
                    mini_batch = replay_memory.sample(self.mini_batch_size)
                    
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    
                    #Epsilon Decay
                    epsilon = max(epsilon * self.epsilon_decay , self.epsilon_min)
                    epsilon_history.append(epsilon)
                    
                    #Copy the policy_network to target_network after step_counter is equal to sync rate
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
    
    
    
    
    
    def save_graph(self, rewards_per_episode, epsilon_history):
        #Save Plots
        fig = plt.figure(figsize=(12,5))
        
        #Plot Avg Rewards V/S Episodes
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99): (x+1)])
        plt.subplot(121)
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)
        
        #Plot Epsilon Decay V/S Episodes
        plt.subplot(122)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)
        
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        
        #Save Plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
        
        
        
        
        
        
        
        
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        #Transpose the list of experiences and separate each element
        states, actions, new_states, rewards ,terminations = zip(*mini_batch)
        
        #Stack tensors to make batch of tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        new_states = torch.stack(new_states)
        terminations = torch.tensor(terminations).float().to(device)
        
        with torch.no_grad():
            if self.enable_double_dqn:
                best__action_from_policy = policy_dqn(new_states).argmax(dim = 1)
            #Calculate target Q values
                target_q = rewards + (1 - terminations) * self.discount_factor * \
                    target_dqn(new_states).gather(dim = 1, index = best__action_from_policy.unsqueeze(dim = 1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor * target_dqn(new_states).max(dim = 1)[0]
        #Calculate the Q values of Policy Network
        current_q = policy_dqn(states).gather(dim = 1, index = actions.unsqueeze(dim = 1)).squeeze()
        #Compute the loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q) 
        
        #Optimize the model
        self.optimizer.zero_grad() #Reset Gradients
        loss.backward()            #Update Gradients using backpropagation
        self.optimizer.step()      #Update network parameters i.e. weights and biases
            
        
              
        
if __name__ == '__main__':
    
    parser  = argparse.ArgumentParser(description="Train or Test Model")
    parser.add_argument("hyperparameters", help='')
    parser.add_argument("--train", help='Training Mode', action='store_true')
    parser.add_argument("--continue-training", help="Continue Training from saved model", action="store_true")
    args = parser.parse_args()
    
    dq1 = Agent(hyperparameter_name=args.hyperparameters)
    
    if args.train:
        dq1.run_agent(is_training=True)
    else:
        dq1.run_agent(is_training=False, render=True)