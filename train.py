import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_module import AnimeDataModule
from main_module import AnimeModule


def main(args):
    seed_everything(1)
    logger = WandbLogger(name=args.description, project="anime2selfie")

    if args.resume == "None":
        path = None
    else:
        path = os.path.join("anime2selfie", args.resume, "checkpoints", "last.ckpt")

    checkpoint = ModelCheckpoint(monitor="g_loss", mode="min", save_last=True)

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger,
        gpus=[args.gpus_id],
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        precision=16,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=path,
        checkpoint_callback=checkpoint,
    )

    data = AnimeDataModule(args)
    model = AnimeModule(args)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--description", type=str, default="Default")
    parser.add_argument("--data_path", type=str, default="/data/huy/selfie2anime")
    parser.add_argument("--prepare_data", type=int, default=0)

    # MODULE specific args
    parser = AnimeModule.add_model_specific_args(parser)

    # DATA specific args
    parser = AnimeDataModule.add_data_specific_args(parser)

    # TRAINER args
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gpus_id", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--no_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--resume", type=str, default="None")
    args = parser.parse_args()
    main(args)
