import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from utils.data_preprocess import make_dataset


class AnimeDataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.hparams = args
        self.train = train
        if self.train:
            splits = ["trainA", "trainB"]
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.Resize(self.hparams.default_img_size + 30),
                    T.RandomCrop(self.hparams.default_img_size),
                    T.Resize(self.hparams.train_img_size),
                ]
            )
        else:
            splits = ["testA", "testB"]
            self.transform = T.Resize(self.hparams.val_img_size)

        cpu = torch.device("cpu")
        self.dataA = torch.load(
            os.path.join(self.hparams.data_path, splits[0] + ".pt"), map_location=cpu
        )
        self.dataB = torch.load(
            os.path.join(self.hparams.data_path, splits[1] + ".pt"), map_location=cpu
        )

        assert self.dataA.size(0) == self.dataB.size(0)

    def __len__(self):
        if self.train:
            return self.dataA.size(0)
        else:
            return self.hparams.no_val_imgs

    def __getitem__(self, index):
        imgA = self.transform(self.dataA[index])
        imgB = self.transform(self.dataB[index])
        return imgA, imgB


class AnimeDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--default_img_size", type=int, default=256)
        parser.add_argument("--train_img_size", type=int, default=256)
        parser.add_argument("--val_img_size", type=int, default=256)
        parser.add_argument("--no_val_imgs", type=int, default=8)
        return parser

    def prepare_data(self):
        if bool(self.hparams.prepare_data):
            print("Preparing data...")
            make_dataset(self.hparams.data_path, "trainA")
            make_dataset(self.hparams.data_path, "trainB")
            make_dataset(self.hparams.data_path, "testA")
            make_dataset(self.hparams.data_path, "testB")
            print("Finished preparing data.")

    def train_dataloader(self):
        dataset = AnimeDataset(self.hparams)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.no_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        dataset = AnimeDataset(self.hparams, train=False)
        return DataLoader(
            dataset,
            batch_size=self.hparams.no_val_imgs,
            num_workers=self.hparams.no_workers,
            pin_memory=True,
        )
