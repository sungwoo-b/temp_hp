import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning as L
from torchvision import models
from timm.layers import NormMlpClassifierHead

from src.loss import FocalLoss


class HPClassifier(L.LightningModule):
    def __init__(
        self,
        lr=1e-3,

        # Update the number of classes for each property
        num_quality=3,
        num_type=4,
        num_soft=2,
        num_story=6,
        num_overhang=3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # backbone = models.resnet18(pretrained=True)
        # in_features = backbone.fc.in_features
        # self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone = timm.create_model(
            "convnextv2_tiny", pretrained=True, drop_path_rate=0.25
        )
        features = 768

        self.quality_head = NormMlpClassifierHead(
            features, num_quality, drop_rate=0.5
        )
        self.type_head = NormMlpClassifierHead(
            features, num_type, drop_rate=0.5
        )
        self.soft_head = NormMlpClassifierHead(
            features, num_soft, drop_rate=0.5
        )
        self.overhang_head = NormMlpClassifierHead(
            features, num_overhang, drop_rate=0.5
        )
        self.story_head = NormMlpClassifierHead(features, num_story, drop_rate=0.5)



        self.quality_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_quality, average="weighted"
        )
        self.type_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_type, average="weighted"
        )
        self.soft_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_soft, average="weighted"
        )
        self.overhang_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_overhang, average="weighted"
        )
        self.story_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_story, average="weighted")
 
        
        


    def forward(self, xb):
        features = self.backbone.forward_features(xb)
        # features = torch.flatten(features, 1)

        quality_logits = self.quality_head(features)
        type_logits = self.type_head(features)
        soft_logits = self.soft_head(features)
        overhang_logits = self.overhang_head(features)
        story_logits = self.story_head(features)

        return (quality_logits, type_logits, soft_logits, overhang_logits, story_logits)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=10, T_mult=2, eta_min=self.hparams.lr * 10, last_epoch=-1
        # )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[8, 22, 36], gamma=0.1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def shared_step(self, batch, phase):

        # Updated the attributes to match our data
        img, quality_idx, type_idx, soft_idx, story_idx, overhang_idx = batch
        (quality_logits, type_logits, soft_logits, overhang_logits, story_logits) = self(img)



        quality_preds = torch.argmax(quality_logits, dim=1)
        type_preds = torch.argmax(type_logits, dim=1)
        soft_preds = torch.argmax(soft_logits, dim=1)
        overhang_preds = torch.argmax(overhang_logits, dim=1)
        story_preds = torch.argmax(story_logits, dim=1) 

        
        # Compute F1 scores for classification tasks
        quality_f1 = self.quality_f1(quality_preds, quality_idx)
        type_f1 = self.type_f1(type_preds, type_idx)
        soft_f1 = self.soft_f1(soft_preds, soft_idx)
        overhang_f1 = self.overhang_f1(overhang_preds, overhang_idx)
        story_f1 = self.story_f1(story_preds, story_idx)


        total_f1 = quality_f1 + type_f1 + soft_f1 + overhang_f1


        quality_loss = F.cross_entropy(quality_logits, quality_idx)
        type_loss = F.cross_entropy(type_logits, type_idx)
        soft_loss = F.cross_entropy(soft_logits, soft_idx)
        overhang_loss = F.cross_entropy(overhang_logits, overhang_idx)
        story_loss = F.cross_entropy(story_logits, story_idx) 

        # When adding Weight, do as follows:

        # complete_loss = F.cross_entropy(
        #     complete_logits,
        #     complete_idx,
        #     weight=torch.as_tensor([0.16, 0.84], device=self.device),
        # )
        # condition_loss = F.cross_entropy(
        #     condition_logits,
        #     condition_idx,
        #     weight=torch.as_tensor([0.19, 0.09, 0.72], device=self.device),
        # )
        # material_loss = F.cross_entropy(
        #     material_logits,
        #     material_idx,
        #     weight=torch.as_tensor(
        #         [0.0046, 0.0017, 0.018, 0.026, 0.196, 0.099, 0.25, 0.39],
        #         device=self.device,
        #     ),
        # )
        # security_loss = F.cross_entropy(
        #     security_logits,
        #     security_idx,
        #     weight=torch.as_tensor([0.75, 0.25], device=self.device),
        # )
        # use_loss = F.cross_entropy(
        #     use_logits,
        #     use_idx,
        #     weight=torch.as_tensor([0.009, 0.759, 0.115, 0.115], device=self.device),
        # )

        loss = (quality_loss + type_loss + soft_loss + overhang_loss + story_loss)    
        # loss = (
        #     complete_loss
        #     + condition_loss
        #     + (material_loss / 4)
        #     + security_loss
        #     + (use_loss / 2)
        # )

        self.log(f"{phase}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{phase}_totalf1", total_f1, prog_bar=True, logger=True)
        
        # self.log(
        #     f"{phase}_loss",
        #     loss,
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        # )

        # # for properties in ("complete", "condition", "material", "security", "use"):
        # self.log(
        #     f"{phase}_totalf1",
        #     total_f1,
        #     prog_bar=True,
        #     logger=True,
        # )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")