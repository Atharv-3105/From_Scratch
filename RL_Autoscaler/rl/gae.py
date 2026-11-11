import numpy as np 

def compute_gae(rewards, values, dones, last_value, gamma, lambd):
    """ 
        generalized advantage estimation, one column per environment time runs down the rows
        rewards, values, dones: (T,N), dones[t] = 1 means the episode ENDED after step t, so nothing after
        it may leak into step t. last_value: (N,) value of the observation after the final step, used to bootstrap
        when the rollout stops in the middle of an episode.
        advantage[t] = delta[t] + gamma * lambda * advantage[t + 1], 
        delta[t] = r[t] + gamma * V[t + 1] - V[t]
        lambda = 0 is one-step TD, lambda = 1 is the full discounted return minus the value
    """
    
    rewards = np.asarray(rewards, dtype = np.float64)
    values = np.asarray(values, dtype = np.float64)
    dones = np.asarray(dones, dtype = np.float64)
    steps, n = rewards.shape 
    
    advantages = np.zeros((steps, n))
    running = np.zeros(n)
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        alive = 1.0 - dones[t]   #will be 0 when the episoded ended at t: 
        delta = rewards[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lambd * alive * running 
        advantages[t] = running 
        
    returns = advantages + values 
    return advantages , returns 