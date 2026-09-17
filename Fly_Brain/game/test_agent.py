from game.flappy import FlappyBirdEnv 
from game.agent import FlyBrainAgent 

env = FlappyBirdEnv(render = True)
agent = FlyBrainAgent()

obs = env.reset()
agent.reset()

episode_reward = 0
steps = 0
max_steps = 300

for step in range(max_steps):
    action, flap_prob = agent.act(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    env.render()
    
    if step % 20 == 0:
        print(f"step[{step}]: flap_prob = {flap_prob:.3f}, action = {action}, reward so far={episode_reward:.2f}")
        
    if done:
        print(f"Episode ended at step:{step} -- score: {info['score']}, total reward: {episode_reward:.2f}")
        break 
    
env.close()