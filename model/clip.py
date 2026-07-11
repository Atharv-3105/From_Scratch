import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    
    def __init__(self, vocab_size:int, n_embed:int, n_tokens:int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        
        #In normal Transformer Architecture , The Positional Encoding is given by Sinosoidal Functions 
        #But in CLIP Architecture, The Positional Embedding/Encoding is given by some parameters which are learned by the model while training.
        self.position_embedding = nn.Parameter(torch.zeros((n_tokens, n_embed)))
        
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        
        x += self.position_embedding
        
        return x 

class CLIPLayer(nn.Module):
    #It is same as how the Encoder is in normal Transformer Architecture{i.e Having }
    def __init__(self, n_head:int, n_embed:int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4*n_embed)
        self.linear_2 = nn.Linear(n_embed*4, n_embed)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # x : (batch_size, seq_len, dim)
        
        residue  = x
        
        #First , We have SelfAttention Connection
        x = self.layernorm_1(x)
        
        x = self.attention(x, causal_mask = True)
        
        x += residue 
        
        #Second, We have Feed Forward Connection
        residue = x 
        
        x = self.layernorm_2(x)
        
        x = self.linear_1(x)
        
        #We use QuickGELU function as our activation function for FeedForward Layer
        x = x*torch.sigmoid(1.702*x)
        
        x = self.linear_2(x)

        x += residue
        
        return x 
    
    
    
    
class CLIP(nn.Module):
    
    def __init__(self):
        #49408:- Vocabulary Size, 768:- Embedding Size, 77:- Max Seq_len
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12) #12 :- Number of Heads in Self-Attention
        ])
        
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens:torch.LongTensor)->torch.FloatTensor:
        tokens  = tokens.type(torch.long)
        
        #(batch_size, seq_len)------>(batch_size, seq_len, dim)
        state = self.embedding(tokens)
        
        #Now the state will transit over each layer present in the class
        for layer in self.layers:
            state = layer(state)
        
        #(batch_size, seq_len, dim)
        #Since, It's a Sequential Model the shape of the input should be same as the shape of the output.
        output = self.layernorm(state)
        
        return output
        
        