import torch
import numpy as np
from tqdm import tqdm 
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = 512 // 8 #LATENT is the input to the Variational AutoEncoder 
LATENT_HEIGHT = 512 // 8

def generate(prompt:str, uncond_prompt:str, input_image=None, strength= 0.8,
            do_cfg= True, cfg_scale = 7.5, sampler_name="ddpm", n_inference_steps = 50,
            models = {}, seed=None, device=None,idle_device=None,  tokenizer = None):
            
            #uncond_prompt:- It's like a negative prompt , meaning the prompt inside this will be considered as Taboo by the model and hence the result will not contain anything related to this prompt
            #do_cfg:- It stands for do classifier free guidance 
            #cfg_score:- It tells our model how much it need to focus on the prompt

        with torch.no_grad():
            
            if not (0 < strength <= 1):
                raise ValueError("Strength must be between 0-1")
            if idle_device:
                to_idle = lambda x:x.to(idle_device)
            else:
                to_idle =  lambda x: x
                
            #Random Number Generator which we will use to generate noise for our images
            generator = torch.Generator(device=device)
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(seed)
            
            clip = models["clip"]
            clip.to(device)
            
            if do_cfg:
                #We will tokenize the conditional prompt using tokenizer
                conditional_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length = 77).input_ids
                
                #Now we convert these tokens into tensors of shape:-(batch_size, seq_len)
                conditional_tokens = torch.tensor(conditional_tokens, dtype=torch.long, device=device)
                
                #Now, we convert these tensor to embeddings of size 768 each which will change shape to:- (batch_size, seq_len, dim{768})
                conditional_context = clip(conditional_tokens)
               
                unconditional_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length = 77).input_ids 
                unconditional_tokens = torch.tensor(unconditional_tokens, dtype=torch.long, device=device)
                #Now, we convert these tensor to embeddings of size 768 each which will change shape to:- (batch_size, seq_len, dim{768})
                unconditional_context = clip(unconditional_tokens)
                
                #Now, we concatenate these 2 contexts which will become the batch of our input to the UNET
                #(2, seq_len, dim) = (2, 77, 768)
                context = torch.cat([conditional_context, unconditional_context])
            else:
                #Convert it into a list of tokens
                tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                #The context in this case will be a ONE BIG tensor
                #(1, seq_len, dim) = (1, 77, 768)
                context = clip(tokens)
            to_idle(clip)
            
            if sampler_name == "ddpm":
                sampler = DDPMSampler(generator)
                sampler.set_inference_timesteps(n_inference_steps)
            else:
                raise ValueError(f"Unkown Sampler Detected:{sampler_name}")
            
            #This is the dimensions of the latent(image) which will be passed in the UNET 
            latent_shape = (1,4,LATENT_HEIGHT, LATENT_WIDTH)
            
            if input_image:
                #If we want to perform IMAGE-to-IMAGE/TEXT type operation
                encoder = models["encoder"]
                encoder.to(device)
                
                input_image_tensor = input_image.resize((WIDTH,HEIGHT))
                input_image_tensor = np.array(input_image_tensor)
                #(Height, Width, Channels{3 i.e RGB})
                input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
                #Since our UNET wants images which have channels in range (-1 to 1) so we will rescale our input_image 
                input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
                #(height,width,channels)-------->(batch_size, height, width , channels)
                input_image_tensor = input_image_tensor.unsqueeze(0)
                #(batch_size, height, width, channels)--------->(batch_size, channels, height,width )
                input_image_tensor = input_image_tensor.permute(0,3,1,2)
                
                encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
                #Now pass the image through the Encoder of our VAE
                latents = encoder(input_image_tensor, encoder_noise)
                
                sampler.set_strength(strength=strength)
                #Now we tell out sampler to add noise to our latent according to the strength defined above.
                latents = sampler.add_noise(latents, sampler.timesteps[0])
                
                to_idle(encoder)

            else:
                #If we want TEXT-to-IMAGE type operation we will give random noise at start to latent
                latents = torch.randn(latent_shape, generator=generator, device=device)
                
                diffusion = models["diffusion"]
                diffusion.to(device)
                
                timesteps = tqdm(sampler.timesteps)
                for i, timestep in enumerate(timesteps):
                    #(1,320)
                    time_embedding = get_time_embedding(timestep).to(device)
                    
                    #(batch_size, 4, latents_height, latents_width)
                    model_input = latents
                    
                    if do_cfg:
                        # (batch_size, 4, latent_heigh, latent_width)---->(2*batch_size, 1*4, 1*latent_height, 1*latent_width)
                        model_input = model_input.repeat(2,1,1,1) #This will basically double the batch_size so that we can have 2 latents which can be used for conditional & unconditional prompts.
                        
                    #Model_Output is basically just the predicted noise by the UNET
                    model_ouput = diffusion(model_input, context, time_embedding)
                    
                    if do_cfg:
                        #If cfg is present then we will split the output of the model in 2 subsequent outputs.
                        output_conditional, output_unconditional = model_ouput.chunk(2)
                        
                        model_ouput = cfg_scale*(output_conditional - output_unconditional) + output_unconditional
                    
                    #Remove noise that is predicted by the UNET
                    latents = sampler.step(timestep, latents, model_ouput)
                        
                to_idle(diffusion)
                
                decoder = models["decoder"]
                decoder.to(device)
                
                images = decoder(latents)
                to_idle(decoder)
                
                #Rescale the image back to 3 channels RGB
                images = rescale(images, (-1,1), (0,255), clamp=True)
                #Inorder to save the images on CPU , we permute the dimensions so that the channel dimension becomes the last dimension.
                #(batch_size, channels, height,width)-------->(batch_size, height, width, channels)
                images = images.permute(0,2,3,1)
                images = images.to("cpu", torch.uint8).numpy()
                return images[0]
            
def rescale(x,old_range,new_range,clamp=False):
    old_min,old_max = old_range
    new_min,new_max = new_range
    x -= old_min
    x *= (new_max - new_min)/(old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x 

def get_time_embedding(timestep):
    #Refer the formula for positional encoding for better understanding
    
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    
    x = torch.tensor([timestep], dtype=torch.float32)[:,None]*freqs[None]
    
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)