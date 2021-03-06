import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_module import AnimeDataModule
from main_module import AnimeModule


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    logger = WandbLogger(name=args.description, project="selfie2anime")

    if args.resume == "None":
        path = None
    else:
        path = os.path.join("selfie2anime", args.resume, "checkpoints", "last.ckpt")

    checkpoint = ModelCheckpoint(
        monitor="g_loss", mode="min", save_top_k=3, save_last=True
    )

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger,
        gpus=-1,
        benchmark=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        resume_from_checkpoint=path,
        checkpoint_callback=checkpoint,
        num_sanity_val_steps=0,
    )

    data = AnimeDataModule(args)
    max_steps = args.max_epochs * len(data.train_dataloader())
    model = AnimeModule(args, max_steps)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--description", type=str, default="Default")
    parser.add_argument("--data_path", type=str, default="/raid/data/huy/selfie2anime")
    parser.add_argument("--prepare_data", type=int, default=0)

    # MODULE specific args
    parser = AnimeModule.add_model_specific_args(parser)

    # DATA specific args
    parser = AnimeDataModule.add_data_specific_args(parser)

    # TRAINER args
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gpu_id", type=str, default="3")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--no_workers", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--resume", type=str, default="None")
    args = parser.parse_args()
    main(args)
