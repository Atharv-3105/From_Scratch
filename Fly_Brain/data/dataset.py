import os 
from neuprint import Client, fetch_neurons, NeuronCriteria as NC, fetch_adjacencies
from dotenv import load_dotenv
import pandas as pd 


load_dotenv()

TOKEN = os.getenv("NEUPRINT_AUTH_TOKEN")
client = Client(
    "https://neuprint.janelia.org",
    dataset = "male-cns:v1.0",
    token = TOKEN
)


print("Connected to: ", client.fetch_version())

sample_neurons, _ = fetch_neurons(NC(status = "Traced"))
print("Total traced neurons found:", len(sample_neurons))
print("\nColumns available: ", list(sample_neurons.columns))

print("\nBreakdown by superclass:")
print(sample_neurons["superclass"].value_counts())

print("\nBreakdown by class (finer-grained):")
print(sample_neurons["class"].value_counts().head(20))

os.makedirs("data", exist_ok= True)

#We will have visual_neurons as our Input neurons as these includes LC/LPLC-type neurons which are well for detecting Looming
#Looming is an approaching object growing larger in the visual field.
visual_proj_neurons, _ = fetch_neurons(NC(superclass="visual_projection", status = "Traced"))

#We will have descending_neurons as our Output neurons as these are the standard "motor decision" makers.
descending_neurons, _ = fetch_neurons(NC(superclass = "descending_neuron", status = "Traced")) 
print(f"Found {len(visual_proj_neurons)} neurons of superclass 'visual_projection'")
print(f"Found {len(descending_neurons)} neurons of superclass 'descending_neurons'")

visual_proj_neurons.to_feather("data/visual_projection_neurons.feather")
descending_neurons.to_feather("data/descending_neurons.feather")

#Check the DIRECT connections between the neurons(it will be small for now as real pathway between neurons runs through multiple intermediate hops)
vp_ids = visual_proj_neurons["bodyId"].tolist()
dn_ids = descending_neurons["bodyId"].tolist()

_, direct_connections = fetch_adjacencies(vp_ids, dn_ids)
print(f"\nDirect visual_projection --> descending_neuron synapses: {len(direct_connections)}")

if len(direct_connections) > 0:
    direct_connections.to_feather("data/visual_to_descending_direct.feather")
    print(direct_connections.sort_values("weight", ascending = False).head(10))