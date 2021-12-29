import torch
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    wrap up the pytorch CIFAR-10 dataset into our own, which will convert images into sequences of integers
    """

    def __init__(self, pt_dataset, clusters, perm=None):
        self.pt_dataset = pt_dataset
        self.clusters = clusters
        self.perm = torch.arange(32 * 32) if perm is None else perm

        self.vocab_size = clusters.size(0)
        self.block_size = 32 * 32 - 1

    def __len__(self):
        return len(self.pt_dataset)

    def __getitem__(self, idx):
        x, y = self.pt_dataset[idx]
        x = torch.from_numpy(np.array(x)).view(-1, 3)  # flatten out all pixels
        x = x[self.perm].float()  # reshuffle pixels with any fixed permutation and -> float
        a = ((x[:, None, :] - self.clusters[None, :, :]) ** 2).sum(-1).argmin(1)  # cluster assignments
        return {'x': a[:-1], 'y': a[1:]}  # always just predict the next one in the sequence