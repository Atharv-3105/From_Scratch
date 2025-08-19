import torch
import torch.nn as nn 
import torch.nn.functional as F 
import threading
import gymnasium as gym
from agent import ActorCritic

class Actors(threading.Thread):
    def __init__(self,global_model, global_optimizer, env_name, actor_id, 
                 global_ep, global_max_score,global_policy_loss, global_value_loss, global_entropy,max_episodes,
                 gamma = 0.99, update_interval = 5):
        super().__init__()
        
        self.global_model = global_model
        self.global_optimizer = global_optimizer
        self.env = gym.make(env_name)
        self.actor_id = actor_id
        self.gamma = gamma
        self.update_interval = update_interval
        self.global_ep = global_ep
        self.global_max_score = global_max_score
        self.global_policy_loss = global_policy_loss
        self.global_value_loss = global_value_loss
        self.global_entropy = global_entropy
        self.max_episodes = max_episodes
        
        
        #Get the Obs_Dim & Action_Dim
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n 
        
        #Create a Local Model for the current Actor
        self.local_model = ActorCritic(obs_dim, action_dim)
        self.local_model.load_state_dict(self.global_model.state_dict())
        
        
    def run(self):
        total_step = 1
        state , _ = self.env.reset()
        
        
        while self.global_ep.value < self.max_episodes:
            ep_reward = 0
            #Define log_probabilites which will be used to calculate Policy loss
            log_probs = []
            
            #Define values to get the state values to calculate the Critic Loss
            values = []
            
            rewards = []
            #Define entropies to store entropies which will be used for exploration
            entropies = []
            
            for _ in range(self.update_interval):
                
                #Convert State to Tensor
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
                #Get the logits and value for current state
                logits, value = self.local_model(state_tensor)
                
                #Calculate the probas and log_probas
                prob = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                
                #Calculate Entropy for current state
                entropy = -(prob * log_prob).sum()
                
                action = torch.multinomial(prob, num_samples=1).item()
                
                #Get the next_state,reward,terminated,truncated,info
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                #Store values of log_probs,values,entropy,reward 
                log_probs.append(log_prob[0, action])
                values.append(value.squeeze())
                rewards.append(reward)
                entropies.append(entropy)
                
                #Update the state to next_state
                ep_reward += reward
                state = next_state
                total_step += 1
                
                if done:
                    #Log Episode Results
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                        ep_num = self.global_ep.value
                        
                    with self.global_max_score.get_lock():
                        
                        #If current Episode Reward is greater than global_max reward 
                        if ep_reward > self.global_max_score.value:
                            self.global_max_score.value = ep_reward
                            
                            #Save Checkpoint when new max_reward is received
                            checkpoint_path = "checkpoints/best_model.pth"
                            torch.save(self.global_model.state_dict(), checkpoint_path)
                            print(f"[Checkpoint] New Max Reward Received: {ep_reward:2f} | Model updated at {checkpoint_path}")
                            
                        print(f"[Actor: {self.actor_id}] | Episode: {ep_num} | Reward: {ep_reward} | Max_Reward: {self.global_max_score.value}")
                    
                    state , _ = self.env.reset()
                    #Reset Ep_reward to 0
                    ep_reward = 0
                    done = False
                    break
            
            #Compute Returns and Advantages
            if done:
                R = 0
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                _ , value = self.local_model(state_tensor)
                R = value.item()
                
                #Apply clipping to the Bootstrap Return
                R = max(min(R, 1e3), -1e3)
                
            returns = []
            
            for r in reversed(rewards):
                
                #Compute the discounted return for each reward in reverse
                R = r + self.gamma *R
                #Since we are recursively computing rewards hence we use insert() instead of append() as R for first time_step will become last if append() is used.
                returns.insert(0, R)
                
            #Convert returns,values,rewards,entropies to Tensor
            
            returns = torch.tensor(returns, dtype=torch.float32)
            rewards = torch.tensor(rewards)
            # rewards = torch.stack(rewards)
            values = torch.stack(values)
            entropies = torch.stack(entropies)
            log_probs = torch.stack(log_probs)
            
            
            #Calculate Advantage
            advantage = returns - values.detach()
            
            if advantage.numel() > 1:
                eps = 1e-8
                std = advantage.std()
                if std < eps:
                    std = eps
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            else:
                advantage = torch.zeros_like(advantage)
            
            #Compute Total Loss
            policy_loss = -(log_probs * advantage.detach()).mean()
            value_loss = advantage.pow(2).mean()
            entropy_loss = entropies.mean()
            
            total_loss = policy_loss + 0.5 * value_loss + 0.5 * entropy_loss
            
            #Log the Losses to Global_loss variables
            with self.global_policy_loss.get_lock():
                self.global_policy_loss.value += policy_loss.item()
            with self.global_value_loss.get_lock():
                self.global_value_loss.value += value_loss.item()
            with self.global_entropy.get_lock():
                self.global_entropy.value += entropy_loss.item()
                
                 
            
            
            #Apply gradients to global model
            self.global_optimizer.zero_grad()
            total_loss.backward()
            #Gradient Clipping for Stable Training 
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=40)
            
            #Synchronize Gradients of Local-Model with Gradients of Global Model
            for local_param , global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                global_param._grad = local_param.grad
                
            self.global_optimizer.step()
            
            #Synchronize the Local_Model with the updated Global_Model
            self.local_model.load_state_dict(self.global_model.state_dict())
            
        
            
            
            
                
            
                
                
                
                