import torch
import torch.nn as nn 

from graph.lif_simulator import FlyBrainSimulator

sim = FlyBrainSimulator()
sim.reset()

encoder = nn.Linear(5, sim.n_vp)

dummy_obs = torch.randn(5, requires_grad=False)
external_current = torch.tanh(encoder(dummy_obs)) * 2.0

dn_spike_total = torch.zeros(sim.n_dn)

for t in range(15):
    sim.step(external_current)
    dn_spike_total = dn_spike_total + sim.get_dn_spikes()
    

loss = dn_spike_total.sum()
loss.backward()

print("Loss value:", loss.item())
print("Encoder weight grad is None:", encoder.weight.grad is None)
print("Encoder weight grad norm:", encoder.weight.grad.norm().item() if encoder.weight.grad is not None else "N/A")
print("Any NaNs in grad:", torch.isnan(encoder.weight.grad).any().item() if encoder.weight.grad is not None else "N/A")