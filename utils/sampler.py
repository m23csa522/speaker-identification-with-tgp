
import torch
from torch.utils.data import Sampler

class FixedLengthBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        random_indices = torch.randperm(len(self.indices)).tolist()
        batches = [random_indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        return iter(batches)

    def __len__(self):
        return len(self.indices) // self.batch_size
