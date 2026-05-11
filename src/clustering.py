import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from src.utils import load_pain_points, save_jsonl

device = "mps" if torch.backends.mps.is_available() else "cpu"

pain_points = load_pain_points("data/processed/r_ciso_pain_points.jsonl")
texts = ["clustering: " + pp["text"] for pp in pain_points]

# https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device)
embeddings = model.encode(texts, show_progress_bar=True)

np.save("data/embeddings/r_ciso_pain_points_embeddings.npy", embeddings)

embeddings_umap = UMAP(n_components=40, metric="cosine", min_dist=0.0).fit_transform(embeddings)

np.save("data/embeddings/r_ciso_pain_points_embeddings_umap.npy", embeddings_umap)

labels = HDBSCAN(min_samples=5, min_cluster_size=100, metric="euclidean").fit_predict(embeddings_umap)

save_jsonl(
    [{**pp, "cluster": int(label)} for pp, label in zip(pain_points, labels)],
    "data/embeddings/r_ciso_pain_points_clustered.jsonl",
)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"{n_clusters} clusters, {int((labels == -1).sum())} noise points")
