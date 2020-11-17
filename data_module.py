import pytorch_lightning as pl
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data_preprocess import make_dataset


class AnimeDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.trainA = torch.load(os.path.join(data_path, "trainA"))
        self.trainB = torch.load(os.path.join(data_path, "trainA"))

        self.img_size = 256
        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.Resize((self.img_size + 30, self.img_size + 30)),
                T.RandomCrop(self.img_size),
            ]
        )

    def __len__(self):
        return self.trainA.size(0)

    def __getitem__(self, index):
        indexB = torch.randint(0, self.trainB.size(0), (1,))
        indexB = indexB.item()

        imgA = self.transform(self.trainA[index])
        imgB = self.transform(self.trainB[indexB])
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
        # make_dataset(self.args.data_path, "testA")
        # make_dataset(self.args.data_path, "testB")

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