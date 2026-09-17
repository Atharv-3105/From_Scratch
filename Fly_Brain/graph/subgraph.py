import os 
import pandas as pd 
import numpy as np 
from neuprint import fetch_neurons, fetch_adjacencies,Client, NeuronCriteria as NC 
from dotenv import load_dotenv


load_dotenv()

TOKEN = os.getenv("NEUPRINT_AUTH_TOKEN")
client = Client(
    "https://neuprint.janelia.org",
    dataset = "male-cns:v1.0",
    token = TOKEN
)


print("Connected to: ", client.fetch_version())

visual_proj_neurons = pd.read_feather("data/visual_projection_neurons.feather")
descending_neurons = pd.read_feather("data/descending_neurons.feather")

vp_ids = set(visual_proj_neurons["bodyId"].tolist())
dn_ids = set(descending_neurons["bodyId"].tolist())

#Step-1 Find neurons ONE HOP upstream of descending neurons and ONE HOP downstream of visual_neurons
_, vp_downstream_conns = fetch_adjacencies(list(vp_ids), None)
downstream_of_vp = set(vp_downstream_conns["bodyId_post"].unique())
_, dn_upstream_conns = fetch_adjacencies(None, list(dn_ids))
upstream_of_dn = set(dn_upstream_conns["bodyId_pre"].unique())

#Step-2 The subgraph = Input pop + Output pop + the overlap/bridge neurons
bridge_candidates = downstream_of_vp.intersection(upstream_of_dn) - vp_ids - dn_ids
print(f"Bridge neuron candidates (1-HOP overlap): {len(bridge_candidates)}")

MAX_BRIDGE_NEURONS = 800

if len(bridge_candidates) > MAX_BRIDGE_NEURONS:
    #Then we RANK bridge neurons by total incoming weight from vp + outgoing weight to dn
    weight_in = vp_downstream_conns[vp_downstream_conns["bodyId_post"].isin(bridge_candidates)].groupby("bodyId_post")["weight"].sum()
    weight_out = dn_upstream_conns[dn_upstream_conns["bodyId_pre"].isin(bridge_candidates)].groupby("bodyId_pre")["weight"].sum()
    
    combined_score = weight_in.add(weight_out, fill_value = 0).sort_values(ascending = False)
    bridge_ids = set(combined_score.head(MAX_BRIDGE_NEURONS).index)
    
else:
    bridge_ids = bridge_candidates
    

all_ids = list(vp_ids | dn_ids | bridge_ids)
print(f"\nFinal Subgraph size: {len(all_ids)} neurons "
      f"({len(vp_ids)} visual_projection + {len(dn_ids)} descending + {len(bridge_ids)} bridge)")

#Step-3 Fetch ALL connections within this final neuron set
neuron_meta, all_connections = fetch_adjacencies(all_ids, all_ids)

print(f"Total connections in subgraph: {len(all_connections)}")

neuron_meta = neuron_meta.reset_index(drop = True)
neuron_meta.to_feather("data/subgraph_neurons.feather")
all_connections.to_feather("data/subgraph_connections.feather")

#Save the role of each neuron(input/output/bridge)
role_lookup = pd.DataFrame({
    "bodyId": all_ids, 
    "role": ["visual_projection" if b in vp_ids else "descending" if b in dn_ids else "bridge" for b in all_ids],
})

# Merge onto neuron_meta's order this is the row order that
# subgraph_neurons.feather will also be saved in, so everything stays aligned.
roles = neuron_meta[["bodyId"]].merge(role_lookup, on="bodyId", how="left")

# Sanity check: every neuron in neuron_meta MUST have gotten a role.
# If this prints > 0, some neurons in neuron_meta weren't in your original
# vp/dn/bridge sets.
print("Unmatched neurons:", roles["role"].isna().sum())

roles = roles.reset_index(drop = True)
roles.to_feather("data/subgraph_roles.feather")
print(roles["role"].value_counts())
