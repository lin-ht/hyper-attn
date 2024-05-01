import nmslib
import numpy as np
import time
import torch
from tqdm import tqdm

class AnnsHNSW(torch.nn.Module):
    # Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graph

    def __init__(self, sample_size, batch_size, head_size):
        super().__init__()
        # Number of neighbors to search for
        self.sample_size = sample_size
        space_name = "l2"

        self.num_threads = 4
        self.index_time_params = {'M': 16, 'indexThreadQty': self.num_threads, 'efConstruction': 400, 'post': 0}
        print('Index-time parameters', self.index_time_params)

        self.key_norm_max = None
        self.index_count = batch_size * head_size
        self.ann_indices = [nmslib.init(method='hnsw', space=space_name) for _ in range(self.index_count)]

    #The Asymmetric QNF transformation:
    # P(k) = [... k_i ..., sqrt(M^2 - ||k||)]
    # Q(q) = [... r*q_i ..., 0], where r=M/||q||
    # M is the maximum norm of the key vectors
    def _transform_key(self, key: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        key_norm = key.norm(dim=-1)
        key_norm_max = key_norm.max(dim=-1, keepdim=True).values
        key_extra = torch.sqrt(key_norm_max ** 2 - key_norm ** 2)
        key_qnf = torch.cat([key, key_extra.unsqueeze_(-1)], dim=-1)
        return key_qnf, key_norm_max

    def _transform_query(self, query: torch.tensor, key_norm_max: torch.tensor) -> torch.tensor:
        query_norm = torch.maximum(query.norm(dim=-1), torch.tensor(1e-6))
        r = key_norm_max / query_norm
        query_qnf = torch.cat([r.unsqueeze_(-1) * query, torch.zeros_like(query_norm).unsqueeze_(-1)], dim=-1)
        return query_qnf

    def build_indices(self, key: torch.tensor):
        key_qnf, key_norm_max = self._transform_key(key)
        self.key_norm_max = key_norm_max

        key_qnf = key_qnf.contiguous()
        key_np = torch.numpy(key_qnf, force=True).reshape(self.index_count, -1, key_qnf.shape[-1])
        tqdm_bar = tqdm(self.ann_indices)
        # for i, index in enumerate(self.ann_indices):
        for i, index in enumerate(tqdm_bar):
            index.addDataPointBatch(key_np[i, :, :])
            index.createIndex(self.index_time_params, print_progress=False)

    def _convert_result(self, neighbors: list[tuple[np.ndarray, np.ndarray]]) -> torch.tensor:
        neighbors_ids = torch.cat([torch.tensor(ids) for ids, _ in neighbors], dim=-1)
        return neighbors_ids

    def anns_samples(self, query: torch.tensor) -> torch.tensor:
        if self.key_norm_max is None:
            raise ValueError("Build the indices first")

        query_qnf = self._transform_key(query, self.key_norm_max)
        query_qnf = query_qnf.contiguous()
        query_np = torch.numpy(query_qnf, force=True).reshape(self.index_count, -1, query_qnf.shape[-1])

        # Get the nearest neighbors
        neighbors = [index.knnQueryBatch(query_np[i, :, :], k=self.sample_size, num_threads=self.num_threads) for i, index in enumerate(self.ann_indices)]
        sampled_set = self._convert_result(neighbors).reshape(*query.shape[:-1], self.sample_size).to(query.device)
        return sampled_set

    def __repr__(self):
        return f"ANNS_HNSW(sample_size={self.sample_size}, index_count={self.index_count})"
