""" 
each neuron has a membrane potential V - think of it as a leaky bucket of electrical charge. 
Incoming spikes from connected neurons add charge; the leak constantly drains it back toward a resting value; 
if V crosses a threshold, the neuron "fires" (emits a spike), resets, and that spike becomes input to whatever it's connected to downstream

The governing equation:
τ · dV/dt = -(V - V_rest) + I(t)
τ (tau) — the membrane time constant. Larger τ = slower leak = the neuron "remembers" recent input longer.
V_rest — resting potential, where V settles with no input.
I(t) — total input current at time t, which is the weighted sum of spikes arriving from upstream neurons this timestep.
"""

import numpy as np 
import pandas as pd 
from scipy import sparse
import torch 
import torch.nn as nn 

from graph.surrogate_spike import surrogate_spike_fn

class FlyBrainSimulator(nn.Module):
    def __init__(self, neurons_path = "data/subgraph_neurons.feather", connections_path = "data/subgraph_connections.feather", roles_path = "data/subgraph_roles.feather",
                 tau = 10.0, v_rest = 0.0, v_reset = 0.0, v_threshold = 1.0, dt = 1.0, w_scale = 0.01, surrogate_alpha = 10.0):
        
        super().__init__()
        

        neurons = pd.read_feather(neurons_path)
        connections = pd.read_feather(connections_path)
        roles = pd.read_feather(roles_path)

        self.body_ids = neurons["bodyId"].tolist()
        self.id_to_idx = {b: i for i, b in enumerate(self.body_ids)}
        self.n_neurons = len(self.body_ids)
        
        def nt_sign(nt):
            if not isinstance(nt, str):
                return 1.0
            nt = nt.lower()
            if nt == "acetylcholine":
                return 1.0
            elif nt in ("gaba", "glutamate"):
                return -1.0
            return 1.0

        neurons["sign"] = neurons["predictedNt"].apply(nt_sign)
        sign_by_idx = neurons.set_index("bodyId").loc[self.body_ids, "sign"].values

        rows = connections["bodyId_pre"].map(self.id_to_idx).values
        cols = connections["bodyId_post"].map(self.id_to_idx).values
        
        weights = connections["weight"].values.astype(np.float32)

        W = sparse.csr_matrix((weights, (rows, cols)), shape=(self.n_neurons, self.n_neurons))
        sign_matrix = sparse.diags(sign_by_idx)
        W_signed = (sign_matrix @ W).tocoo()
        
        # We need W^T for the recurrent step: result[j] = sum_i spikes[i] * W[i, j]
        # torch.sparse.mm(A, b) computes A @ b, so store the TRANSPOSE directly.
        W_T = W_signed.transpose().tocoo()
        indices = torch.tensor(np.vstack([W_T.row, W_T.col]), dtype=torch.long)
        values = torch.tensor(W_T.data, dtype=torch.float32)

        # register_buffer: this tensor moves with the model (e.g. .to(device)),
        # gets saved/loaded, but is NEVER updated by the optimizer — this IS
        # the "frozen connectome" constraint enforced at the PyTorch level.
        self.register_buffer("W_T_sparse_indices", indices, persistent=True)
        self.register_buffer("W_T_sparse_values", values, persistent=True)
        self.W_shape = (self.n_neurons, self.n_neurons)
        self._W_T_cached = None   #built lazingly once, on first use 
        
        # Masks for input/output populations — computed once, reused every step
        vp_mask = (roles["role"] == "visual_projection").values
        dn_mask = (roles["role"] == "descending").values
        
        self.register_buffer("vp_mask", torch.tensor(vp_mask, dtype = torch.bool))
        self.register_buffer("dn_mask", torch.tensor(dn_mask, dtype = torch.bool))
        self.n_vp = int(vp_mask.sum())
        self.n_dn = int(dn_mask.sum())

        self.tau = tau
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.dt = dt
        self.w_scale = w_scale
        self.surrogate_alpha = surrogate_alpha

        self.V = None 
        self.spikes = None
        
    def _W_T(self):
        if self._W_T_cached is None:
            coo = torch.sparse_coo_tensor(self.W_T_sparse_indices, self.W_T_sparse_values, self.W_shape).coalesce()
            self._W_T_cached = coo.to_sparse_csr()
        return self._W_T_cached
        
    def reset(self, device = "cpu"):
        """
            Call this at the start of every game episode — resets neural state,
            NOT the connectome structure (that stays frozen forever).
        """
        self.V = torch.zeros(self.n_neurons, device = device)
        self.spikes = torch.zeros(self.n_neurons, device = device)

    def step(self, external_input_vp):
        """
            external_input_vp: torch tensor, shape (n_vp,), REQUIRES GRAD (comes from encoder)
            Returns: spikes, shape (n_neurons,) — differentiable via the surrogate function
        """
        external = torch.zeros(self.n_neurons, device = self.V.device)
        external[self.vp_mask] = external_input_vp

        W_T = self._W_T()
        
        # sparse @ dense -> shape (n_neurons, 1) -> squeeze to (n_neurons,)
        recurrent_input = torch.sparse.mm(W_T, self.spikes.unsqueeze(1)).squeeze(1) * self.w_scale
    
        total_input = recurrent_input + external

        dV = (self.dt / self.tau) * (-(self.V - self.v_rest) + total_input)
        V_new = self.V + dV
        
        spikes_new = surrogate_spike_fn(V_new, self.v_threshold, self.surrogate_alpha)

        # Reset: where spikes fired, pull V back to v_reset.
        # Written this way (not in-place indexing) so it stays differentiable.
        V_new = V_new * (1.0 - spikes_new) + self.v_reset * spikes_new

        self.V = V_new
        self.spikes = spikes_new
        return spikes_new
    
    def get_dn_spikes(self):
        """ 
            Read current descending-neuron spike rate, what our decoder will consumer
        """
        return self.spikes[self.dn_mask]
        
        