import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Callable
from scipy.spatial.distance import cdist
from scipy.spatial import distance
import math
from collections import Counter
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize_bm25(text):
    # Basic tokenization: lowercase and split by spaces (you can improve this with regex or other libraries)
    return text.lower().split()

def get_bm25_scores(query, docs):
    query_tokens = tokenize_bm25(query)
    doc_tokens = [tokenize_bm25(doc) for doc in docs]
    bm25 = BM25Okapi(doc_tokens)
    return bm25.get_scores(query_tokens)


def rank_rrf(dense_rank, bm25_rank, k=60):
    final_rank = []
    for dr, bm in zip(dense_rank, bm25_rank):
        final_rank.append(1/(k + dr) + 1/(k + bm))
    return final_rank


class EpisodicMemory:
    def __init__(self, args):
        """Initializes an empty episodic memory."""
        self.memory = []
        self.args = args

    def write(self, sentence: str, embedding: np.ndarray, steering_vector: np.ndarray, reward: float):
        """
        Inserts a new experience tuple into memory.

        Args:
            sentence (str): The text sentence.
            embedding (np.ndarray): The embedding of the sentence.
            steering_vector (np.ndarray): The steering vector associated with the sentence.
            reward (float): The reward associated with this experience.
        """
        experience = {
            "sentence": sentence,
            "embedding": embedding,
            "steering_vector": steering_vector,
            "reward": reward
        }
        self.memory.append(experience)

    def read(self, args, query_embedding: np.ndarray, query_text, distance_metric, k: int = 5) -> List[Tuple[str, torch.Tensor, torch.Tensor, float]]:
            """
            Retrieves the top-k nearest neighbors from memory based on a distance metric.
            """
            if not self.memory:
                return []

            # Extract embeddings from memory
            # self.consider_memory = self.memory[1:]
            embeddings = torch.stack([entry["embedding"].squeeze(0) for entry in self.memory])

            # NOTE: we add negative sign in front of the distance to get the closest distances at the end
            distances = []
            true_distances = []
            if distance_metric == "cosine":
                for i, emb in enumerate(embeddings):
                    d = distance.cosine(query_embedding, emb.cpu().numpy())
                    distances.append(-d)
                    true_distances.append(d)
            elif distance_metric == "euclidean":
                for emb in embeddings:                    
                    d = distance.euclidean(query_embedding, emb.cpu().numpy())
                    distances.append(-d)
                    # round up to 2 decimal places
                    true_distances.append(round(d, 2))
            else:
                raise ValueError

            # rank all the distances by euclidean/ cosine distances
            sorted_indices = np.argsort(true_distances)

            # Get the indices corresponding to the original positions
            dense_rank = np.argsort(sorted_indices)

            scores = get_bm25_scores(query_text, [self.memory[idx]["sentence"] for idx in range(len(self.memory))])
            sorted_indices_bm25 = np.argsort(scores)
            bm25_rank = np.argsort(sorted_indices_bm25)
            final_rank = rank_rrf(dense_rank, bm25_rank)

            # get the top k smallest ranks from the final rank
            top_k_indices = np.argsort(final_rank)[:k]
            
            steering_vector = torch.mean(
                torch.stack([self.memory[idx]["steering_vector"] for idx in top_k_indices]), 
                dim=0
            )

            # calculate uncertainty
            uncertainty = (np.mean(true_distances) - np.min(true_distances)) / (np.max(true_distances) - np.min(true_distances))
            
            return steering_vector, uncertainty


    def __len__(self):
        return len(self.memory)
    

