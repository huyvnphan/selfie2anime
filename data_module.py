import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data_preprocess import make_dataset


class AnimeDataset(Dataset):
    def __init__(self, data_path, train=True):
        super().__init__()
        self.img_size = 256

        if self.train:
            splits = ["trainA", "trainB"]
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.Resize((self.img_size + 30, self.img_size + 30)),
                    T.RandomCrop(self.img_size),
                ]
            )
        else:
            splits = ["testA", "testB"]
            self.transform = T.CenterCrop(self.img_size)

        self.dataA = torch.load(os.path.join(data_path, splits[0]))
        self.dataB = torch.load(os.path.join(data_path, splits[1]))

        assert self.dataA.size(0) == self.dataB.size(0)

    def __len__(self):
        if self.train:
            return self.dataA.size(0)
        else:
            return 8

    def __getitem__(self, index):
        imgA = self.transform(self.dataA[index])
        imgB = self.transform(self.dataB[index])
        return imgA, imgB


class AnimeDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def setup(self):
        make_dataset(self.data_path, "trainA")
        make_dataset(self.data_path, "trainB")
        make_dataset(self.data_path, "testA")
        make_dataset(self.data_path, "testB")

    def train_dataloader(self):
        dataset = AnimeDataset(self.data_path)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.no_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = AnimeDataset(self.data_path, train=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.no_workers,
            shuffle=False,
            pin_memory=True,
        )
