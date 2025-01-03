"""Generate embeddings from knowledge graph."""

import os
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import utils

# Read config file
config = utils.get_config("./config.yaml")

# Create all paths if they dont exist
for path in config["paths"]:
    if not os.path.exists(config["paths"][path]):
        os.makedirs(config["paths"][path])

edges = pd.read_csv(os.path.join(config["paths"]["raw_data"], "Edges.csv"))
nodes = pd.read_csv(os.path.join(config["paths"]["raw_data"], "Nodes.csv"))

# Subset the nodes and edges to avoid OutOfMemory error
nodes = nodes[["id"]]
edges = edges[["subject", "object"]]
nodes = nodes[:1000]
edges = edges[:10000]

# Create a graph
G = nx.Graph()
G.add_nodes_from(nodes["id"])
G.add_edges_from(edges[["subject", "object"]].itertuples(index=False, name=None))

# Use Node2Vec to generate embeddings. Parameters adjusted to allow faster computation.
node2vec = Node2Vec(G, dimensions=32, walk_length=10, num_walks=100, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}
embeddings_df = pd.DataFrame(
    list(embeddings.items()), columns=["id", "topological_embedding"]
)

# Save embeddings dataframe
embeddings_df.to_csv(
    os.path.join(config["paths"]["processed_data"], "generated_embeddings.csv")
)
