""" 
    Since spike = (V >= threshold) is non-differentiable function meaning its derivaties are zero everywhere
    except an undefined spike at the threshold itseld.Meaning it's useless for GRADIENT DESCENT
    
    So we add a Surrogate-Trick meaning we add our own forward, backware function to mimic how real gradients will flow
    
    We use a real, exact step function during the forward pass so our simulation behaves like genuine biological spiking,
    and we substitue a smooth, fake derivative during the backware pass.
    
    The surrogate derivative we'll use is the fast sigmoid surrogate:
    surrogate_grad = 1 / (1 + α·|V - threshold|)²
    This peaks at 1.0 exactly at the threshold (where a real neuron is "on the verge" of firing that's where we most want gradient signal to flow) 
    and decays smoothly to near-zero far from threshold (where nudging weights wouldn't plausibly change whether that neuron fires).
    alpha controls how sharply it decays — 
    higher alpha = surrogate hugs the real step function more closely, but gives less gradient signal to work with; 
    lower α = smoother/more forgiving gradients, less biologically faithful.
"""
import torch

class SurrogateSpike(torch.autograd.Function):
    """
    Forward: exact Heaviside step (real spiking behavior, unchanged from Brick 1).
    Backward: smooth fast-sigmoid surrogate derivative, enabling gradient flow.
    """
    @staticmethod 
    def forward(ctx, V, threshold, alpha):
        ctx.save_for_backward(V, torch.tensor(threshold), torch.tensor(alpha))
        return (V >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        V, threshold, alpha = ctx.saved_tensors
        surrogate_grad = 1.0 / (1.0 + alpha * torch.abs(V - threshold)) ** 2
        grad_input = grad_output * surrogate_grad
        #Only V gets a gradient, threshold and alpha are fixed hyperparametes
        return grad_input, None, None 
    
    
surrogate_spike_fn = SurrogateSpike.apply



