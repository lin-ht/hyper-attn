import torch


class AngularLSH(torch.nn.Module):

    def __init__(self, num_projs, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs

        if num_projs > 0:
            # proj_dir is a tensor of shape (*dim, num_projs)
            self.register_buffer('proj_dir', torch.randn(dim + (num_projs,), generator=rng), persistent=False)
            # Fixme: this seems to be a bug, perm should be a mapping from hash code to angular hash index.
            # perm is the angular hamming code sequence arranged in order
            # self.register_buffer('perm', self._unit_hamming_distance_array(self.num_projs), persistent=False)
            self.register_buffer('perm', self._hamming_code_to_order_mapping_perm(self.num_projs), persistent=False)
            # self.register_buffer('perm', torch.randperm(2 ** num_projs), persistent=False)
            # Example: num_projs=4, enc_vec=[[[[1, 2, 4, 8]]]]
            self.register_buffer('enc_vec', 2 ** torch.arange(self.num_projs).view(1, 1, 1, -1), persistent=False)

    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1])
        a = self._unit_hamming_distance_array(size_n - 1)
        # Examples size_n=1: [0, 1], 2: [0, 1, 3, 2], 3: [0, 1, 3, 2, 6, 7, 5, 4]
        # i.e. 1: [0, 1], 2: [00, 01, 11, 10], 3: [000, 001, 011, 010, 110, 111, 101, 100]
        # Hamming code length is size_n which is able to mark 2 ** size_n elements.
        return torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)

    # Fixme: use perm = self._hamming_code_to_order_mapping(num_projs)
    def _hamming_code_to_order_mapping(self, size_n):
        hamming_codes_in_order = self._unit_hamming_distance_array(size_n)
        # Map hamming code to hash id:
        # Example: size_n=3,
        # hamming_codes_in_order = [0, 1, 3, 2, 6, 7, 5, 4]
        # hamming_codes_indices  = [0, 1, 3, 2, 7, 6, 4, 5]
        return torch.argsort(hamming_codes_in_order)

    # Directly construct the hamming code to angular hash index mapping.
    def _hamming_code_to_order_mapping_perm(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1])

        a = self._hamming_code_to_order_mapping_perm(size_n - 1)
        a_len = len(a)  # a_len = 2 ** (size_n - 1)
        a_len_half = a_len // 2
        return torch.concat([a, a[a_len_half:] + a_len, a[:a_len_half] + a_len], 0)

    def hash(self, mat):
        if self.num_projs < 0:
            return torch.zeros(mat.shape[:-1], device=mat.device, dtype=torch.int32)
        # Feature vector along d is projected to the hyperplane defined by each projection vector
        mask = torch.einsum('...nd,...dr -> ...nr', mat, self.proj_dir)
        mask = mask > 0  # mask is the hamming code in binary form
        bin_ids = (mask * self.enc_vec).sum(-1)  # bin_ids is the hamming code in integer form
        # Random index for our testing case.
        # bin_ids = torch.randint(2 ** self.num_projs, size=bin_ids.shape, device=bin_ids.device)
        # Ground truth index for our testing case.
        return torch.arange(mat.shape[-2], device=bin_ids.device).repeat(mat.shape[0], mat.shape[1], 1)
        return self.perm[bin_ids]  # map hamming code to hash index (in angular order)

    def __repr__(self):
        return f"AngularLSH(num_proj={self.num_projs}, proj_dir.shape={self.proj_dir.shape})"
