import itertools
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn

from networks import Discriminator, ResnetGenerator, RhoClipper


class AnimeModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.save_hyperparameters(args)

        # Define Generator, Discriminator
        self.genA2B = ResnetGenerator()
        self.genB2A = ResnetGenerator()
        self.disGA = Discriminator(n_layers=7)
        self.disGB = Discriminator(n_layers=7)
        self.disLA = Discriminator(n_layers=5)
        self.disLB = Discriminator(n_layers=5)

        # Define Loss
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()

        # Define Rho clipper to constraint the value of rho in AdaILN and ILN
        self.Rho_clipper = RhoClipper(0, 1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--adv_weight", type=float, default=1.0)
        parser.add_argument("--cycle_weight", type=float, default=10.0)
        parser.add_argument("--identity_weight", type=float, default=10.0)
        parser.add_argument("--cam_weight", type=float, default=1000.0)
        return parser

    def training_step(self, batch, batch_nb):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx):
        pass

    def configure_optimizers(self):
        G_optim = torch.optim.Adam(
            itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
            weight_decay=self.weight_decay,
        )
        D_optim = torch.optim.Adam(
            itertools.chain(
                self.disGA.parameters(),
                self.disGB.parameters(),
                self.disLA.parameters(),
                self.disLB.parameters(),
            ),
            lr=self.lr,
            betas=(0.5, 0.999),
            weight_decay=self.hparams.weight_decay,
        )
        G_scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                G_optim, self.hparams.max_steps
            ),
            "interval": "step",
        }
        D_scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                D_optim, self.hparams.max_steps
            ),
            "interval": "step",
        }

        return [G_optim, D_optim], [G_scheduler, D_scheduler]
