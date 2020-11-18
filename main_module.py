import itertools
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import AdamW

import wandb
from networks import Discriminator, ResnetGenerator, RhoClipper


class AnimeModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.save_hyperparameters(args)

        # Define Generator, Discriminator
        self.genA2B = ResnetGenerator(light=bool(self.hparams.light_model))
        self.genB2A = ResnetGenerator(light=bool(self.hparams.light_model))
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
        parser.add_argument("--light_model", type=int, default=0, choices=[0, 1])
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--adv_weight", type=float, default=1.0)
        parser.add_argument("--cycle_weight", type=float, default=10.0)
        parser.add_argument("--identity_weight", type=float, default=10.0)
        parser.add_argument("--cam_weight", type=float, default=1000.0)
        return parser

    def forward(self, x):
        # Transfer real images to anime version
        anime, _, _ = self.genA2B(x)
        return anime

    def discriminator_loss(self, real_A, real_B):
        with torch.no_grad():
            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A.detach())
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A.detach())
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B.detach())
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B.detach())

        D_ad_loss_GA = self.MSE_loss(
            real_GA_logit, torch.ones_like(real_GA_logit)
        ) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit))
        D_ad_cam_loss_GA = self.MSE_loss(
            real_GA_cam_logit, torch.ones_like(real_GA_cam_logit)
        ) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit))
        D_ad_loss_LA = self.MSE_loss(
            real_LA_logit, torch.ones_like(real_LA_logit)
        ) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit))
        D_ad_cam_loss_LA = self.MSE_loss(
            real_LA_cam_logit, torch.ones_like(real_LA_cam_logit)
        ) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit))
        D_ad_loss_GB = self.MSE_loss(
            real_GB_logit, torch.ones_like(real_GB_logit)
        ) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
        D_ad_cam_loss_GB = self.MSE_loss(
            real_GB_cam_logit, torch.ones_like(real_GB_cam_logit)
        ) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit))
        D_ad_loss_LB = self.MSE_loss(
            real_LB_logit, torch.ones_like(real_LB_logit)
        ) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))
        D_ad_cam_loss_LB = self.MSE_loss(
            real_LB_cam_logit, torch.ones_like(real_LB_cam_logit)
        ) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit))

        D_loss_A = self.hparams.adv_weight * (
            D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA
        )
        D_loss_B = self.hparams.adv_weight * (
            D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB
        )

        Discriminator_loss = D_loss_A + D_loss_B
        return Discriminator_loss

    def generator_loss(self, real_A, real_B):
        fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

        fake_A2B2A, _, _ = self.genB2A(fake_A2B)
        fake_B2A2B, _, _ = self.genA2B(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit))
        G_ad_cam_loss_GA = self.MSE_loss(
            fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit)
        )
        G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit))
        G_ad_cam_loss_LA = self.MSE_loss(
            fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit)
        )
        G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit))
        G_ad_cam_loss_GB = self.MSE_loss(
            fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit)
        )
        G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit))
        G_ad_cam_loss_LB = self.MSE_loss(
            fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit)
        )

        G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
        G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

        G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
        G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

        G_cam_loss_A = self.BCE_loss(
            fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit)
        ) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit))
        G_cam_loss_B = self.BCE_loss(
            fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit)
        ) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit))

        G_loss_A = (
            self.hparams.adv_weight
            * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA)
            + self.hparams.cycle_weight * G_recon_loss_A
            + self.hparams.identity_weight * G_identity_loss_A
            + self.hparams.cam_weight * G_cam_loss_A
        )
        G_loss_B = (
            self.hparams.adv_weight
            * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB)
            + self.hparams.cycle_weight * G_recon_loss_B
            + self.hparams.identity_weight * G_identity_loss_B
            + self.hparams.cam_weight * G_cam_loss_B
        )

        Generator_loss = G_loss_A + G_loss_B
        return Generator_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        if optimizer_idx == 0:
            # clip parameter of AdaILN and ILN
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            d_loss = self.discriminator_loss(real_A, real_B)
            self.log("d_loss", d_loss, on_epoch=False, prog_bar=True)
            return d_loss
        else:
            # use the generator loss for checkpointing
            g_loss = self.generator_loss(real_A, real_B)
            self.log("g_loss", g_loss, on_epoch=False, prog_bar=True)
            return g_loss

    def validation_step(self, batch, batch_idx):
        realA, _ = batch
        anime = make_grid(self(realA))

        anime_img_grid = [
            wandb.Image(anime, caption="Epoch_" + str(self.current_epoch))
        ]
        self.logger.experiment.log({"Anime_Image": anime_img_grid})

        if self.current_epoch == 0:
            real_img_grid = [wandb.Image(realA, caption="Real_Image")]
            self.logger.experiment.log({"Real_Image": real_img_grid})

    def configure_optimizers(self):
        D_optim = AdamW(
            itertools.chain(
                self.disGA.parameters(),
                self.disGB.parameters(),
                self.disLA.parameters(),
                self.disLB.parameters(),
            ),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
            weight_decay=self.hparams.weight_decay,
        )
        G_optim = AdamW(
            itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        reduce_lr = [
            int(0.50 * self.hparams.max_epochs),
            int(0.75 * self.hparams.max_epochs),
            int(0.90 * self.hparams.max_epochs),
        ]
        D_scheduler = {
            "scheduler": MultiStepLR(D_optim, milestones=reduce_lr),
            "interval": "epoch",
        }
        G_scheduler = {
            "scheduler": MultiStepLR(G_optim, milestones=reduce_lr),
            "interval": "epoch",
        }
        return [D_optim, G_optim], [D_scheduler, G_scheduler]
