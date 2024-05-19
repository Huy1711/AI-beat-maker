import pytorch_lightning as pl
import torch
import torch_optimizer as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..criterion.contrastive_loss import NTxentLoss
from ..data.datasets import MusicSegmentDataset, collate_data
from ..model.neuralfp import NeuralAudioFingerprinter


class AudioFingerprint(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = NeuralAudioFingerprinter(**config["model"]["neuralfp"])
        self.criterion = NTxentLoss()

    def _shared_step(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        norm=True,
    ) -> torch.Tensor:
        # torch.stack([x, x_aug], dim=0) => [x1_orig, x2_orig, ..., x1_aug, x2_aug ...]
        xs = torch.stack([xs, ys], dim=0)
        xs = torch.flatten(xs, 0, 1)
        out = self.model(xs, norm=norm)
        n_anchors = out.shape[0] // 2
        loss = self.criterion(out[:n_anchors, :], out[n_anchors:, :], n_anchors)
        return loss

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        loss = self._shared_step(xs, ys)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        loss = self._shared_step(xs, ys)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def train_dataloader(self) -> DataLoader:
        train_ds = MusicSegmentDataset(config=self.config["dataset"]["train"])
        train_dl = DataLoader(
            dataset=train_ds,
            collate_fn=collate_data,
            shuffle=True,
            **self.config["dataset"]["loaders"],
        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        val_ds = MusicSegmentDataset(config=self.config["dataset"]["val"])
        val_dl = DataLoader(
            dataset=val_ds,
            collate_fn=collate_data,
            shuffle=False,
            **self.config["dataset"]["loaders"],
        )
        return val_dl

    def configure_optimizers(self):
        optimizer = optim.Lamb(
            self.parameters(),
            **self.config["optimizer"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(self.train_dataloader()),
            **self.config["scheduler"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def export(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "model": self.model.state_dict(),
            },
            "hyper_parameters": self.hparams,
        }
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint has been saved to "{filepath}"')
