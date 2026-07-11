import torch
import numpy as np

class DDPMSampler:
    
    def __init__(self, generator:torch.Generator, num_training_steps =1000, beta_start = 0.00085, beta_end = 0.0120):
        
        #torch.linspace(start,end,steps):- Generates a 1D tensor of evenly space values between start and end.
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        
        #torch.cumprod():- Computes the cumulative product of elements along the given dimension.
        self.alpha_cumprod = torch.cumprod(self.alphas,dim=0)
        self.one = torch.tensor(1.0)
        
        self.generator = generator
        self.num_training_steps = num_training_steps
        #Since the arange() will give 0-1000 number we reverse it as we require 1000-0 format.
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        
    def set_inference_timesteps(self, num_inference_steps = 50):
        self.num_inference_steps = num_inference_steps
        #For inference steps = 1000 { 999 , 998 , 997 , 996 ,.....}
        #For inference steps = 50   {999, 999-20, 999-40, 999-60,.....}
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0,num_inference_steps)*step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
    
    def get_previous_timestep(self, timestep:int) -> int:
        prev_timestep = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_timestep
    
    def get_variance(self, timestep:int)->torch.Tensor:
        prev_t = self.get_previous_timestep(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >=0  else self.one
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        
        current_beta_t = 1 - alpha_prod_t / alpha_prod_prev_t
        variance = (beta_prod_prev_t / beta_prod_t)*current_beta_t
        
        #torch.clamp():- It's a function which is used to clamp/bound a tensor values between MIN & MAX range.
        variance = torch.clamp(variance, min=1e-20)
        
        return variance
    
            
    def step(self, timestep:int, latents:torch.Tensor, model_output:torch.Tensor):
        t = timestep
        prev_t = self.get_previous_timestep(t)   

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t   
        
        #Compute the original sample using the formula(15) of the DDOM Paper
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        #Compute the co-efficients for pred_original sample and current sample x_t
        pred_original_sample_coeff = (alpha_prod_prev_t ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_prev_t / beta_prod_t
        
        #Compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff* latents
        
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator , device=device, dtype=model_output.dtype)
            variance = (self.get_variance(t) ** 0.5)*noise
            
        #N(0,1)---->N(mu, sigma^2) = N(mean , std_deviation^2)
        #X = mu + sigma*Z where Z is N(0,1) 
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample
            
            
    def set_strength(self, strength = 1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        #We redefine the timesteps as we don't want the UNET to work on 100% noise image at the start of IMAGE-to-IMAGE process 
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        
    
    
    
    
    def add_noise(self, original_samples:torch.FloatTensor, timesteps: torch.IntTensor)->torch.FloatTensor:
        #q(Xt|X0) = N(Xt;(_@t)**0.5 x X0,(1 - _@t)I)
        #Where; @t = alpha for time t {@ = 1 - beta} and _@t = alpha_bar for time t is the cumulative product of each alpha
        
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            #This will keep adding dimension to our Alpha bar until it is of the same shape of original_sample
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5 #Standard Deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        
        while(len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape)):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype = original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
            
            