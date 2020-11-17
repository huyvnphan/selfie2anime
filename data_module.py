import pytorch_lightning as pl
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data_preprocess import make_dataset

img_size = 256
train_transform = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.Resize((img_size + 30, img_size + 30)),
        T.RandomCrop(img_size),
    ]
)


class AnimeDataset(Dataset):
    def __init__(self, data_path, split):
        super().__init__()
        self.all_imgs = torch.load(os.path.join(data_path, split))

    def __len__(self):
        return self.all_imgs.size(0)

    def __getitem__(self, index):
        img = self.all_imgs[index]
        img = train_transform(img)
        return img


class Concat_Dataloaders:
    def __init__(self, loaderA, loaderB):
        self.loaderA = loaderA
        self.loaderB = loaderB
        self.iterA = iter(self.loaderA)
        self.iterB = iter(self.loaderB)

    def __iter__(self):
        self.loader_iter = [iter(self.loaderA), iter(self.loaderB)]
        return self

    def __next__(self):
        try:
            imgA = self.iterA.next()
        except StopIteration:
            self.iterA = iter(self.loaderA)
            imgA = self.iterA.next()

        try:
            imgB = self.iterB.next()
        except StopIteration:
            self.iterB = iter(self.loaderB)
            imgB = self.iterB.next()

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
        trainA = AnimeDataset(self.data_path, "trainA")
        trainB = AnimeDataset(self.data_path, "trainB")

        trainA_loader = DataLoader(
            trainA,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        trainB_loader = DataLoader(
            trainB,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

        train_loader = Concat_Dataloaders(trainA_loader, trainB_loader)
        return train_loader
