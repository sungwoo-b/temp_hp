from pathlib import Path

from einops import rearrange
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning as L
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Fixed to fit our data
class HouseDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        self.quality = {"informal": 0, "medium": 1, "high": 2}
        self.type = {"residential": 0, "commercial": 1, "mixed": 2, "industrial": 3}
        self.soft = {"yes": 0, "no": 1}
        self.story = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        self.overhang = {"none": 0, "structural": 1, "non-structural": 2}

        # Need to check weights later!

        #self.weights = self.df["weights"]
        #self.weights = self.df.get("weights", None)

        if "weights" in df.columns:
            self.weights = df["weights"].values
        else:
            self.weights = None  # Set to None if weights column is not found


    def __getattr__(self, name):
        "Creates a reverse mapping for all the building properties"
        if name.startswith("r") and hasattr(self, name[1:]):
            _attr = getattr(self, name[1:])
            if isinstance(_attr, dict):
                return {v: k for k, v in _attr.items()}
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row.file_name
        img = read_image(str(img_path))

        # Convert string labels to integers using the mappings

        # Fixed to fit our data
        quality = self.quality[row.quality]
        type = self.type[row.type]
        soft = self.soft[row.soft]
        story = self.story[row.story]
        overhang = self.overhang[row.overhang]

        if self.transform:
            img = self.transform(img)

        return img, quality, type, soft, story, overhang

    def show(self, idx):
        # Fixed to fit our data
        img, quality, type, soft, story, overhang = self[idx]
        title = f"{self.rquality[quality]} | {self.rtype[type]} | {self.rsoft[soft]} | {self.rstory[story]} | {self.roverhang[overhang]}"
        img = rearrange(img, "c h w -> h w c")
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)


def resize_and_pad(target_width, target_height):
    def transform(image):
        # Original dimensions
        _, original_width, original_height = image.shape

        # Determine scaling factor and resize
        scaling_factor = min(
            target_width / original_width, target_height / original_height
        )
        new_width, new_height = int(original_width * scaling_factor), int(
            original_height * scaling_factor
        )
        resized_image = v2.Resize((new_height, new_width), antialias=True)(image)

        # Calculate padding
        padding_left = (target_width - new_width) // 2
        padding_top = (target_height - new_height) // 2
        padding_right = target_width - new_width - padding_left
        padding_bottom = target_height - new_height - padding_top

        # Pad the resized image
        padded_image = v2.Pad(
            (padding_left, padding_top, padding_right, padding_bottom),
            fill=0,
            padding_mode="constant",
        )(resized_image)

        return padded_image

    return transform


class HouseDataModule(L.LightningDataModule):
    def __init__(self, img_dir, data_dir, batch_size, num_workers):
        self.img_dir = Path(img_dir)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trn_tfm = v2.Compose(
            [
                # resize_and_pad(224, 224),
                #v2.RandomResizedCrop(224, scale=(0.8, 1.2), antialias=True),
                v2.RandomResizedCrop(512, scale=(0.8, 1.2), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(
                    degrees=(0, 30),
                    translate=(0.1, 0.3),
                ),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.val_tfm = self.tst_tfm = v2.Compose(
            [
                # resize_and_pad(224, 224),
                #v2.Resize((224, 224), antialias=True),
                v2.Resize((512, 512), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            trn_df = pd.read_csv(self.data_dir / f"train.csv")
            val_df = pd.read_csv(self.data_dir / f"valid.csv")

            self.trn_ds = HouseDataset(trn_df, self.img_dir, self.trn_tfm)

            # We do not have the weights in our data
            if self.trn_ds.weights is not None:
                self.trn_sampler = WeightedRandomSampler(
                    weights=self.trn_ds.weights,
                    num_samples=len(self.trn_ds),
                    replacement=True,
                )
            else:
                self.trn_sampler = None

            # self.trn_sampler = WeightedRandomSampler(
            #     weights=self.trn_ds.weights,
            #     num_samples=len(self.trn_ds),
            #     replacement=True,
            # )

            self.val_ds = HouseDataset(val_df, self.img_dir, self.val_tfm)
            
            tst_df = pd.read_csv(self.data_dir / f"test.csv")
            self.tst_ds = HouseDataset(tst_df, self.img_dir, self.tst_tfm)
            
        
        elif stage == "test" or stage is None:
            tst_df = pd.read_csv(self.data_dir / f"test.csv")
            self.tst_ds = HouseDataset(tst_df, self.img_dir, self.tst_tfm)
            #self.tst_ds = (self.data_dir, train=False)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            sampler=self.trn_sampler,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tst_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )