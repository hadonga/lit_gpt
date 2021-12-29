import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision
import torch
import numpy as np

from data.cifar_dataset import ImageDataset

class CifarDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(CifarDataModule, self).__init__()
        self.args=args

    def prepare_data(self):
        self.train_data = torchvision.datasets.CIFAR10(self.args.dataset_dir, train=True, transform=None, target_transform=None,
                                                  download=True)
        self.test_data = torchvision.datasets.CIFAR10(self.args.dataset_dir, train=False, transform=None, target_transform=None,
                                                 download=True)

        # run kmeans to get codebook self.C
        pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32 * 32, 3)[torch.randperm(32 * 32)[:5], :]
        px = torch.cat([pluck_rgb(x) for x, y in self.train_data], dim=0).float()
        print(px.size())
        ncluster = 512
        with torch.no_grad():
            self.C = self.kmeans(px, ncluster, niter=8)

        self.vocab_size= self.C.size(0)
        self.block_size= 32*32 -1
        self.train_data_length= len(self.train_data)

        # return {"vocab_size": vocab_size, 'block_size': block_size}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageDataset(self.train_data, self.C)
            self.test_dataset = ImageDataset(self.test_data, self.C)
            print(self.train_data[0][0])
        if stage == 'test':
            pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                          pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                          pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return None


    def kmeans(self, x, ncluster, niter=10):
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]]  # init clusters at random
        for i in range(niter):
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print('done step %d/%d, re-initialized %d dead clusters' % (i + 1, niter, ndead))
            c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
        return c
