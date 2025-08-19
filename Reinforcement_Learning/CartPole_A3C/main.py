import torch
import gymnasium as gym
import torch.multiprocessing as mp 
import torch.nn as nn 
from agent import ActorCritic
from actor import Actors
from utils import evaluate, plot_results, record_video
import matplotlib.pyplot as plt
import time 
import os

if __name__ == "__main__":
    #Create the environment
    env = gym.make("CartPole-v1")

    #Get the Observation Dim from the Environment
    obs_dim = env.observation_space.shape[0]

    #Get the Action Dim from the Environment
    action_dim = env.action_space.n

    #Close the Environment 
    env.close

    # Create a Global Model, which will be accessed by all the actors and hence it should  be in shared_memory
    global_model = ActorCritic(obs_dim, action_dim)
    #Allow shared_access of Model 
    global_model.share_memory()

    #Create a Global Optimizer, which will be used by all the actors;
    # It's gradients will be updated by actors and applied to Global Model
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)
    
    #Define the Number of Actors 
    num_actors = 1
    
    
    #Define a Shared_Variable which will be used for tracking
    global_ep = mp.Value('i', 0)  #Integer for Episode_count
    global_max_score = mp.Value('d', 0.0) #Double for max_reward obtained in each episode
    global_policy_loss = mp.Value('d', 0.0) #Global Variable to store Policy Loss value
    global_value_loss = mp.Value('d', 0.0)  #Global Variable to store Value Loss value
    global_entropy = mp.Value('d', 0.0)   #Global Variable to store Entropy values
     
    actors = []
    eval_interval = 50
    max_episodes = 1000
    
    for actor_id in range(num_actors):
        actor = Actors(
            global_model=global_model,
            global_optimizer=global_optimizer,
            env_name="CartPole-v1",
            actor_id=actor_id,
            global_ep = global_ep,
            global_max_score = global_max_score,
            global_policy_loss=global_policy_loss,
            global_value_loss=global_value_loss,
            global_entropy=global_entropy,
            max_episodes=max_episodes
        )
        actor.start()
        actors.append(actor)
        
    scores = []
    max_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    
    
    
    #Evaluation Loop
    while True:
        if global_ep.value >= max_episodes:
            break
        
        if global_ep.value % eval_interval == 0:
            
            #Do Agent Evaluation
            avg_rewards = evaluate(global_model, env_name = "CartPole-v1")
            max_reward = global_max_score.value
            
            avg_policy_loss = global_policy_loss.value / num_actors
            avg_value_loss = global_value_loss.value / num_actors
            avg_entropy = global_entropy.value / num_actors
            
            global_policy_loss.value = 0.0
            global_value_loss.value = 0.0
            global_entropy.value = 0.0
            
            print("##################<Evaluating Agent>##################")
            print(f"Ep {global_ep.value} | Avg_Reward: {avg_rewards:.3f} | Max Reward: {max_reward:.3f} | "
                  f"Policy_Loss: {avg_policy_loss:.3f} | Value_Loss: {avg_value_loss:.3f} | Entropy: {avg_entropy:.3f}")
            
            scores.append((global_ep.value, avg_rewards))
            max_rewards.append((global_ep.value, max_reward))
            policy_losses.append((global_ep.value, avg_policy_loss))
            value_losses.append((global_ep.value, avg_value_loss))
            entropies.append((global_ep.value, avg_entropy))
            
            time.sleep(1)
            
    for actor in actors:
        actor.join()
        
    #Save Model
    model_save_path = "saves/model.pth"
    torch.save(global_model.state_dict(), model_save_path)
    print(f"Model State Dict saved successfully to {model_save_path}")
    
    #Generate Plots
    plot_results(scores=scores,
                 max_rewards=max_rewards,
                 policy_losses=policy_losses,
                 value_losses=value_losses,
                 entropies=entropies,
                 save_path=model_save_path)
    
    #Record Videos
    record_video(global_model=global_model,
                 env_name="CartPole-v1",
                 video_folder=model_save_path)
    