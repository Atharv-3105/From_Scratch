import torch
from torch import nn as nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock,VAE_ResidualBlock


#Encoder Class Architecture
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(batch_size, channel , height , width)------>(batch_size, 128, height, width)
            nn.Conv2d(3,128, kernel_size=3, padding=1),
            
            #ResidualBlock will have combinations of Convolutions & Normalizations
            #It will not change the size of the Image
            #(batch_size, 128, height, width)----->(batch_size, 128, height, width) 
            VAE_ResidualBlock(128,128),
            
            #(batch_size, 128, height, width)----->(batch_size, 128, height, width) 
            VAE_ResidualBlock(128,128),
            
            #(batch_size, 128, height , width)--------->(batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            #(batch_size, 128, height/2, width/2)----->(batch_size, 256, height/2, width/2) 
            VAE_ResidualBlock(128,256),
            
            #The image size is getting reduced but the Number of Features is getting increased
            
            #(batch_size, 128, height/2, width/2)----->(batch_size, 256, height/2, width/2) 
            VAE_ResidualBlock(256,256),
            
            #(batch_size, 256, height/2, width/2)------>(batch_size, 256, height/4, width/4)
            nn.Conv2d(256,256, kernel_size=3, stride=2, padding=0),
            
            #(batch_size, 256, height/4, width/4)------->(batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256,512),
            
            #(batch_size, 512, height/4, width/4)------->(batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512,512),
            
            #(batch_size, 512, height/4, width/4)------->(batch_size, 512, height/8, width/8)
            nn.Conv2d(512,512,kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(512,512),
            
            VAE_ResidualBlock(512,512),
            
            #(batch_size, 512, height/8, width/8)-------->(batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512,512),
            
            #Goal of AttentionBlock:- It is used to relate the pixels of image to each other
            #Just like in TEXT data AttentionBlock is used to relate text tokens with each other
            #(batch_size, 512, height/8, width/8)--------->(batch_size, 512, height/8, width/8)
            VAE_AttentionBlock(512),
            
            #(batch_size, 512, height/8, width/8)------->(batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512,512),
            
            #(batch_size, 512, height/8, width/8)-------->(batch_size, 512, height/8 , width/8)
            nn.GroupNorm(32, 512),
            
            #SiLU layer(Sigmoid Linear Unit)
            nn.SiLU(),
            
            #(batch_size, 512, height/8 , width/8)-------->(batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            #(batch_size, 8, height/8 , width/8)-------->(batch_size, 8 , height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
            
        )
        
    def forward(self, x:torch.Tensor , noise:torch.Tensor)->torch.Tensor:
        #x : (batch_size, channels, height, width)
        #noise: It has dimensions same as that of the Output of the encoder (batch_size, Out_Channels, height/8 , width/ 8)
        
        for module in self:
            #We need to apply Special Embed Padding to Modules having stride as attribute
            if getattr(module, 'stride', None) == (2,2):
                #(Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0,1,0,1))
            x = module(x)
        
        #(batch_size, 8, height/ 8, width/8)------>2 tensors of shape (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2 , dim=1)  #torch.chunk:- It will divide the tensor into chunks(i.e. 2)
        
        #(batch_size, 4, height/8, width/8)-------->(batch_size, 4, height/8, width/8)
        log_variance = torch.clamp(log_variance, -30, 20) # It's a  function that limits(clamps) the values of a tensor to a specified range.
        
        #(batch_size, 4, height/8, width/8)-------->(batch_size, 4, heigth/8, width/8)
        variance = log_variance.exp() #It's a function that computes the exponential of each element in a tensor.
        
        std_deviation = variance.sqrt()   
        
        x = mean + std_deviation* noise 
        
        #Scale the output by a constant 
        x *= 0.18125
        
        return x 