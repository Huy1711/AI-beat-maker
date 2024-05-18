import argparse

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

from neuralfp.module.audio_fingerprint import AudioFingerprint


def main(config: DictConfig):

    module = AudioFingerprint(config)

    if config.model.get("pretrained_weights") is not None:
        checkpoint = config.task.model.pretrained_weights
        print(f"Load checkpoint at {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location="cpu")
        for attr, weights in checkpoint["state_dict"].items():
            net = getattr(module, attr)
            net.load_state_dict(weights, strict=False)

    callbacks = []
    learning_rate_cb = pl.callbacks.LearningRateMonitor()
    callbacks.append(learning_rate_cb)
    if config.callbacks.get("cb"):
        checkpoint_cb = pl.callbacks.ModelCheckpoint(**config.callbacks.checkpointing)
        callbacks.append(checkpoint_cb)

    loggers = []
    if config.loggers.get("tensorboard"):
        tb_logger = pl.loggers.TensorBoardLogger(**config.loggers.tensorboard)
        loggers.append(tb_logger)

    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        required=False,
        default="configs/train.yaml",
        help='path to training config file (.yaml, .yml)'
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config)
