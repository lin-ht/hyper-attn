import nmslib
import numpy as np
import time
import torch
from tqdm import tqdm

class AnnsHNSW(torch.nn.Module):
    # Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graph

    def __init__(self, sample_size):
        super().__init__()
        # Number of neighbors to search for
        self.sample_size = sample_size
        self.space_name = "l2"

        self.num_threads = 4
        self.index_time_params = {'M': 16, 'indexThreadQty': self.num_threads, 'efConstruction': 400, 'post': 0}
        print('Index-time parameters', self.index_time_params)

        self.key_norm_max = None
        self.index_count = 0
        self.k_anns_indices = []
        # self.q_anns_indices = []

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

    def build_indices(self, query: torch.tensor, key: torch.tensor) -> None:
        batch_size, head_size = key.shape[:2]
        self.index_count = batch_size * head_size

        key_qnf, key_norm_max = self._transform_key(key)
        key_qnf = key_qnf.contiguous()
        self.key_np = torch.numpy(key_qnf, force=True).reshape(self.index_count, -1, key_qnf.shape[-1])
        self.key_norm_max = key_norm_max

        query_qnf = self._transform_query(query, key_norm_max)
        query_qnf = query_qnf.contiguous()
        self.query_np = torch.numpy(query_qnf, force=True).reshape(self.index_count, -1, query_qnf.shape[-1])

        tqdm_bar = tqdm(range(self.index_count), desc="Building ANNS indices")
        # for i, index in enumerate(self.ann_indices):
        self.k_anns_indices = []
        for i in tqdm_bar:
            k_anns_index = nmslib.init(method='hnsw', space=self.space_name)
            k_anns_index.addDataPointBatch(self.key_np[i, :, :])
            k_anns_index.createIndex(self.index_time_params, print_progress=False)
            self.k_anns_indices.append(k_anns_index)

        # self.q_anns_indices = []
        # for i in tqdm_bar:
        #     q_anns_index = nmslib.init(method='hnsw', space=self.space_name)
        #     q_anns_index.addDataPointBatch(self.query_np[i, :, :])
        #     q_anns_index.createIndex(self.index_time_params, print_progress=False)
        #     self.q_anns_indices.append(q_anns_index)


    def _convert_result(self, neighbors: list[tuple[np.ndarray, np.ndarray]]) -> torch.tensor:
        neighbors_ids = torch.cat([torch.tensor(ids) for ids, _ in neighbors], dim=-1)
        return neighbors_ids

    def anns_pairing(self, query: torch.tensor, key: torch.tensor) -> torch.tensor:
        if self.key_norm_max is None:
            print("Building the indices first")
            self.build_indices(query, key)

        batch_size, head_size, n_query = query.shape[:3]

        # Get k (sample_size) nearest neighbors for each query.
        neighbors = [index.knnQueryBatch(self.query_np[i, :, :], k=self.sample_size, num_threads=self.num_threads) for i, index in enumerate(self.k_anns_indices)]
        sampled_set = self._convert_result(neighbors).reshape(*query.shape[:-1], self.sample_size).to(query.device)
        # Queries are now labeled with their k nearest neighbors.
        # Use only the 1st nearest neighbor for query segmentation/clustering.
        label_set = sampled_set[..., 0]
        # TODO: use linear gathering under label instead of quicksort.
        # Pick all the 1st nearest neighbors of each k queries.
        key_pick_idx_1st, query_sort_idx = torch.sort(label_set, dim=-1)  # query_sort_idx: [batch_size, head_size, n_query]
        # Sort the sampled_set according to the query_sort_idx.
        sampled_set_sorted = torch.gather(sampled_set, dim=-2, index=query_sort_idx.unsqueeze(-1).expand(-1, -1, -1, self.sample_size))
        sampled_set_picked = sampled_set_sorted[:, :, ::self.sample_size, :]
        # TODO: Refine the key_pick_idx_1st by replacing duplicates within each k queries with the 2nd or other nearest neighbors.
        key_pick_idx = sampled_set_picked.to(device=query.device).contiguous().reshape([batch_size, head_size, -1])[:, :, :n_query]
        return query_sort_idx, key_pick_idx

    def __repr__(self):
        return f"ANNS_HNSW(sample_size={self.sample_size}, index_count={self.index_count})"
