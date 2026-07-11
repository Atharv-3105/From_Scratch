import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    
    def __init__(self,  n_heads:int, d_embed:int, in_proj_bias =True, out_proj_bias=True):
        super().__init__()
        
        #Instead of making 3 matrices {Q,K,V} we will define one big matrix
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x:torch.Tensor , causal_mask = False):
        # x: (batch_size, seq_len, dim)
        
        input_shape = x.shape
        batch_size, sequence_length , d_embed = input_shape
        
        
        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 
        
        #We are splitting the big matrix into 3 smaller matrices 
        #(batch_size, seq_len, dim)----->(batch_size, seq_len, dim*3)----> 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        #(batch_size, seq_len, dim)----->(batch_size, seq_len, num_heads, dim/head)---->(batch_size, num_heads, seq_len, dim/heads)
        #Each head will watch over the whole sequence
        q = q.view(intermediate_shape).transpose(1,2)
        k = k.view(intermediate_shape).transpose(1,2)
        v = v.view(intermediate_shape).transpose(1,2)
        
        #(batch_size, num_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1,-2)
        
        if causal_mask:
            #This is a mask where the upper triangle of the Principal Diagonal has to be given values 1 
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        #(batch_size, num_heads, seq_len, seq_len) @( batch_size, num_heads , seq_len, dim/heads)---->(batch_size, num_heads, seq_len, dim/h)
        output = weight @ v
        
        #(batch_size, num_heads , seq_len, dim/h)--------->(batch_size, seq_len, num_heads,dim/h)
        output = output.transpose(1,2 )

        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        return output 
    
class CrossAttention(nn.Module):
    
    def __init__(self, n_head:int, d_embed:int, d_cross:int, in_proj_bias=True, out_proj_bias = True):
        super().__init__()
        
        #In CrossAttention we will define 3 seperate matrices for (q,k,v) instead of defining 1 big matrix for (q,k,v) like in SelfAttention
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias) 
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)   
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_head = n_head
        self.d_head = d_embed//n_head    #How much information each head will see     
        
    def forward(self, x, y):
        # x is (query)(latent):- (batch_size, seq_len_q , dim_q)
        # y is (key,values)(prompt):- (batch_size, seq_len_kv, dim_kv) = (batch_size, 77, 768) 
        
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_head, self.d_head)
        
        #Multiply Q matrix by Wq 
        q = self.q_proj(x)
        
        #Multiply K matrix by Wk
        k = self.k_proj(y)
        
        #Multiply V matrix by Wv
        v = self.v_proj(y)
        
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)
        
        weight = q @ k.transpose(-1,-2)
        
        weight = weight / math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        
        output = output.transpose(1,2).contiguous()
        
        output = output.view(input_shape)
        
        output = self.out_proj(output)
        
        return output