import pygame 
import numpy as np 
import random 

class FlappyBirdEnv:
    def __init__(self, width = 400, height = 600, render = True):
        self.width = width 
        self.height = height 
        self.render_enabled = render 
        
        #Bird Physics
        self.bird_x = 80 
        self.bird_radius = 12 
        self.gravity = 0.5
        self.flaps_strength = -8.0
        self.max_fall_speed = 10.0
        
        #Pipe physics
        self.pipe_width = 60
        self.pipe_gap = 160
        self.pipe_speed = 4.0
        self.pipe_spawn_distance = 220 #Horizontal gap between successive pipes 
        
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Fly Brain Flappy Bird")
            
        self.reset()
        
    def reset(self):
        self.bird_y = self.height / 2
        self.bird_vy = 0.0
        
        #Each pipe: {"x": ...., "gap_y": ...}
        self.pipes = []
        self._spawn_pipe(x = self.width + 100)
        self._spawn_pipe(x = self.width + 100 + self.pipe_spawn_distance)
        
        self.done = False 
        self.score = 0
        return self._get_obs()
    
    def _spawn_pipe(self, x):
        gap_y = random.uniform(self.pipe_gap, self.height - self.pipe_gap)
        self.pipes.append({"x": x, "gap_y": gap_y, "scored": False})
        
    def _next_pipe(self):
        #The nearest pipe still ahead of(or at) the bird's x position 
        ahead = [p for p in self.pipes if p["x"] + self.pipe_width >= self.bird_x]
        return min(ahead, key = lambda p: p["x"]) if ahead else self.pipes[0]
    
    def _get_obs(self):
        pipe = self._next_pipe()
        
        # Horizontal distance to the pipe this IS the looming signal.
        # As this shrinks toward zero, the "object" is approaching.
        dist_x = (pipe["x"] - self.bird_x) / self.width
        
        #vertical offset between bird and the center of the gap it needs to fly through
        gap_offset_y = (pipe["gap_y"] - self.bird_y) / self.height
        
        return np.array([
            self.bird_y / self.height * 2 - 1,   #This is the bird vertical position, normalized
            self.bird_vy / self.max_fall_speed,  #This is the bird's vertical velocity
            dist_x,                              #Distance to the next pipe(looming cue)
            gap_offset_y,                        #Where the gap is relative to the bird
            self.pipe_gap / self.height,         #Gap size 
        ], dtype = np.float32)
    
    
    def step(self, action):
        """ 
            Action: 1 = flap, 0 = do nothing
        """
        if action == 1:
            self.bird_vy = self.flaps_strength
        else:
            self.bird_vy = min(self.bird_vy + self.gravity, self.max_fall_speed)
        
        self.bird_y += self.bird_vy
        
        #Move pipes left
        for pipe in self.pipes:
            pipe["x"] -= self.pipe_speed
            
        #Spawn new pipe once the leftmost has scrolled far enough 
        if self.pipes[0]["x"] < -self.pipe_width:
            self.pipes.pop(0)
            last_x = self.pipes[-1]["x"]
            self._spawn_pipe(x = last_x + self.pipe_spawn_distance)
        
        reward = 0.05   #Small per-frame survival reward encourages staying alive
        self.done = False 
        
        if self.bird_y - self.bird_radius <= 0 or self.bird_y + self.bird_radius >= self.height:
            reward = -1.0
            self.done = True 
            
        #Check collision with the nearest pipe
        pipe = self._next_pipe()
        bird_left = self.bird_x - self.bird_radius
        bird_right = self.bird_x + self.bird_radius
        pipe_left = pipe["x"]
        pipe_right = pipe["x"] + self.pipe_width
        
        if bird_right > pipe_left and bird_left < pipe_right:
            gap_top = pipe["gap_y"] - self.pipe_gap / 2
            gap_bottom = pipe["gap_y"] + self.pipe_gap / 2
            if not(gap_top < self.bird_y - self.bird_radius and self.bird_y + self.bird_radius < gap_bottom):
                reward = -1.0
                self.done = True 
                
        #Score when the bird passes a pipe cleanly
        if not pipe["scored"] and pipe["x"] + self.pipe_width < self.bird_x:
            pipe["scored"] = True 
            self.score += 1 
            reward = 1.0
        
        obs = self._get_obs()
        return obs, reward, self.done, {"score": self.score}
    
    def render(self):
        if not self.render_enabled:
            return 
        
        self.screen.fill((30, 30, 60))
        
        for pipe in self.pipes:
            gap_top = pipe["gap_y"] - self.pipe_gap / 2
            gap_bottom = pipe["gap_y"] + self.pipe_gap / 2
            pygame.draw.rect(self.screen, (0, 180, 0), (pipe["x"], 0, self.pipe_width, gap_top))
            pygame.draw.rect(self.screen, (0, 180, 0),(pipe["x"], gap_bottom, self.pipe_width, self.height - gap_bottom))
            
        pygame.draw.circle(self.screen, (255, 220, 0), (self.bird_x, int(self.bird_y)), self.bird_radius)
        
        pygame.display.flip()
        self.clock.tick(60)
        
    def close(self):
        if self.render_enabled:
            pygame.quit()
            
# ---------------------------
# Scripted policy (sanity check only)
# ---------------------------

def scripted_policy(obs):
    """
    Flap if we're below the gap center, otherwise let gravity pull down.
    Purely to verify the environment logic — no learning here.
    """
    gap_offset_y = obs[3]
    return 1 if gap_offset_y < -0.03 else 0


def random_policy(obs):
    return random.choice([0, 0, 1])  # biased toward "do nothing" like real flap timing

# ---------------------------
# Test loop
# ---------------------------

if __name__ == "__main__":
    env = FlappyBirdEnv(render=True)
    obs = env.reset()

    episode_reward = 0
    episodes = 0
    max_episodes = 10

    running = True
    while running and episodes < max_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = scripted_policy(obs)  # swap to random_policy(obs) to compare     
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        env.render()

        if done:
            print(f"Episode {episodes+1} finished — score: {info['score']}, total reward: {episode_reward:.2f}")
            episodes += 1
            obs = env.reset()
            episode_reward = 0

    env.close()