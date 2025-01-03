"""Prepare embeddings and combine to create into features and target."""

import numpy as np
import pandas as pd
import os
import utils


def map_embeddings_to_ground_truth(
    ground_truth_df,
    embeddings_df,
    col_to_map,
    id_col="id",
    embedding_col="topological_embedding",
):
    mapped_embeddings = ground_truth_df.merge(
        embeddings_df[[id_col, embedding_col]],
        left_on=col_to_map,
        right_on=id_col,
        how="left",
    )
    mapped_embeddings = mapped_embeddings.rename(
        columns={embedding_col: col_to_map + "_embedding"}
    )
    mapped_embeddings = mapped_embeddings.drop(columns=[id_col])

    return mapped_embeddings


# Read config file
config = utils.get_config("./config.yaml")

# Read in ground truth data and embedding data for model training and evaluation
embeddings = pd.read_csv(os.path.join(config["paths"]["raw_data"], "Embeddings.csv"))
ground_truth = pd.read_csv(
    os.path.join(config["paths"]["raw_data"], "Ground Truth.csv")
)
ground_truth = ground_truth.drop(columns=["Unnamed: 0"])

# Create new columns with source and target embeddings mapped from embeddings dataframe
mapped_embeddings = map_embeddings_to_ground_truth(
    ground_truth, embeddings, col_to_map="source"
)
mapped_embeddings = map_embeddings_to_ground_truth(
    mapped_embeddings, embeddings, col_to_map="target"
)

# Convert embeddings to numpy arrays to allow combining of embeddings into one array
for col in ["source_embedding", "target_embedding"]:
    mapped_embeddings[col] = mapped_embeddings[col].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
source_embeddings = np.array(mapped_embeddings["source_embedding"].tolist())
target_embeddings = np.array(mapped_embeddings["target_embedding"].tolist())
combined_embeddings = np.hstack((source_embeddings, target_embeddings))

# Get model features and target
features = pd.DataFrame(combined_embeddings)
target = mapped_embeddings["y"]

# Save processed data
features.to_pickle(os.path.join(config["paths"]["processed_data"], "features.pkl"))
target.to_pickle(os.path.join(config["paths"]["processed_data"], "target.pkl"))
