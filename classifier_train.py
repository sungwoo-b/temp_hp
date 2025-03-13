import sys
import torch
#from aim.pytorch_lightning import AimLogger
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Using this logger for now since AimLogger is not supported on Windows
from lightning.pytorch.loggers import TensorBoardLogger

from src.datamodule import HouseDataModule
from src.model import HPClassifier
from src.callbacks import BackboneFreezeUnfreeze

# Add Memory Release Callback Class
class ClearCudaCacheCallback(L.Callback):
    def on_batch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

L.seed_everything(42, workers=True)


def main(name, img_dir, data_dir):

    # loggers

    logger = TensorBoardLogger("logs/", name=name)
    #logger = None
    # logger = AimLogger(
    #     experiment=name,
    #     train_metric_prefix="train_",
    #     val_metric_prefix="val_",
    # )
    # datamodule
    dm = HouseDataModule(
        img_dir=img_dir,
        data_dir=data_dir,
        batch_size=16, # 128, Reduced batch size to 16 due to memory constraints
        num_workers=8,
    )
    dm.setup()

    # model
    model = HPClassifier(lr=1e-3)

    # Callbacks
    lr_cb = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )
    ckpt_cb = ModelCheckpoint(
        monitor="val_totalf1",
        mode="max",
        save_top_k=2,
        save_last=True,
        verbose=True,
        filename="epoch{epoch}_step{step}_loss{val_loss:.3f}_f1{val_totalf1:.3f}",
        auto_insert_metric_name=False,
    )
    backbone_freeze_unfreeze_cb = BackboneFreezeUnfreeze(unfreeze_at_epoch=10)
    clear_cache_cb = ClearCudaCacheCallback()  # Added memory release callback

    # trainer
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=50,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[lr_cb, ckpt_cb, backbone_freeze_unfreeze_cb, clear_cache_cb], # Added memory release callback
    )

    # fit
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    # test
    trainer.test(
        ckpt_path="best",
        dataloaders=dm.test_dataloader(),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classifier_train.py <EXPERIMENT_NAME> <IMG_DIR> <DATA_DIR>")
        sys.exit(1)
    EXPERIMENT_NAME = sys.argv[1]
    IMG_DIR = sys.argv[2]
    DATA_DIR = sys.argv[3] # where the partitioned csvs are
    main(EXPERIMENT_NAME, IMG_DIR, DATA_DIR)