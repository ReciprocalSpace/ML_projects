from collections import OrderedDict
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm.notebook import tqdm

import cv2 as cv
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split

!pip install -U segmentation-models-pytorch
import segmentation_models_pytorch as smp
from torchvision import transforms
from torchvision import models
from PIL import Image


from sklearn import metrics

import matplotlib.pyplot as plt



root = Path("../input/mridata/")
path_db = Path("../input/mridata/database.csv")
path_model_checkpoint = Path("./model_checkpoint.pt")
path_best_model = Path("./model_best.pt")
# Hyperparameter selection

# NN Model
dropout = 0.1

# Dataset
batch_size = 32

# optimizer : Adam
lr = 0.0002
betas = (0.9, 0.99)

# Algorithm
epochs = 25


class MRIDataset(Dataset):
    def __init__(self, csv_file_path, root_dir):
        # Saves the entire database : it is relatively small
        self.mri_database = pd.read_csv(csv_file_path)
        self.images = []
        self.segmentation_masks = []
        self.labels = []

        for i in self.mri_database.index:
            image_path = root_dir / self.mri_database["image_path"][i]
            mask_path = root_dir / self.mri_database["mask_path"][i]
            label = self.mri_database["mask"][i]

            self.images.append(Image.open(image_path))
            self.segmentation_masks.append(Image.open(mask_path))
            self.labels.append(label)

        self.labels = torch.tensor(self.labels)

        self.root_dir = root_dir

        self.resize = transforms.Resize((128, 128))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.mri_database)

    def __getitem__(self, item):
        image = self.images[item]
        mask = self.segmentation_masks[item]
        label = self.labels[item]

        image = self.resize(image)
        mask = self.resize(mask)

        rotation = (2 * np.random.rand() - 1) * 20
        translate_x = np.round((2 * np.random.rand() - 1) * 0.05 * 224)
        translate_y = np.round((2 * np.random.rand() - 1) * 0.05 * 224)

        image = transforms.functional.affine(image, rotation, (translate_x, translate_y), 1, 0)
        mask = transforms.functional.affine(mask, rotation, (translate_x, translate_y), 1, 0)

        flip = np.random.rand() > 0.5
        if flip:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        image = self.normalize(self.to_tensor(image))
        mask = self.to_tensor(mask)
        mask = torch.round(mask)

        return image, mask, label

    def show(self, item):
        im, mk, label = self[item]
        im = im.permute(1, 2, 0).numpy()
        mk = mk.permute(1, 2, 0).numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        im = std * im + mean
        im = np.clip(im, 0, 1)

        im = np.where(mk, np.array([[255, 0, 0]]), im)

        plt.imshow(im)
        plt.title(f"Label: {'tumor' if label else 'no tumor'}")
        plt.axis('off')
        plt.show()

mri_dataset = MRIDataset(path_db, root)
mri_dataset.show(5)


class DoubleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bloc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.bloc(x)


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.double_conv_layer = DoubleConvLayer(in_channels, out_channels)
        self.maxpool_layer = nn.MaxPool2d((2, 2))
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        skip = self.double_conv_layer(x)
        down = self.maxpool_layer(skip)
        down = self.dropout_layer(down)
        return down, skip


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.double_conv_layer = DoubleConvLayer(in_channels, out_channels)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = self.double_conv_layer(x)
        out = self.dropout_layer(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.up_layer = nn.ConvTranspose2d(in_channels - out_channels, in_channels - out_channels, kernel_size=2,
                                           stride=2)
        self.double_conv_layer = DoubleConvLayer(in_channels, out_channels)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, down, skip):
        up = self.up_layer(down)
        up = torch.cat([up, skip], dim=1)
        up = self.double_conv_layer(up)
        up = self.dropout_layer(up)
        return up


class SegUNet(nn.Module):
    def __init__(self, dropout=0.1):
        """
        """
        super(SegUNet, self).__init__()
        self.encoder_1 = EncoderLayer(3, 64, dropout)
        self.encoder_2 = EncoderLayer(64, 128, dropout)
        self.encoder_3 = EncoderLayer(128, 256, dropout)

        self.bottleneck = BottleneckLayer(256, 512, dropout)

        self.decoder_3 = DecoderLayer(512 + 256, 256, dropout)
        self.decoder_2 = DecoderLayer(256 + 128, 128, dropout)
        self.decoder_1 = DecoderLayer(128 + 64, 64, dropout)

        self.final_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.output_layer = nn.Sigmoid()

    def forward(self, x):
        x, skip1 = self.encoder_1(x)
        x, skip2 = self.encoder_2(x)
        x, skip3 = self.encoder_3(x)
        x = self.bottleneck(x)
        x = self.decoder_3(x, skip3)
        x = self.decoder_2(x, skip2)
        x = self.decoder_1(x, skip1)
        x = self.final_layer(x)
        return self.output_layer(x)


class NNTrainer:
    #     model, optimizer, criterion, device, lr_scheduler, path_model_checkpoint, False
    def __init__(self, model, optimizer, criterion, device, lr_scheduler=None,
                 path_best_model: Path = None, path_model_checkpoint: Path = None, from_checkpoint=True):
        self.path_model_checkpoint = path_model_checkpoint
        self.best_model_path = path_best_model
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler

        self.epoch = 0

        self.loss = []

        self.metrics = []
        self.iou = smp.utils.metrics.IoU(threshold=0.5)

        # Check if model already exists
        if from_checkpoint:
            if path_model_checkpoint.is_file():
                self.load_checkpoint()

    def save_checkpoint(self):
        checkpoint_data = {'epoch': self.epoch,
                           'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': self.optimizer.state_dict(),
                           'loss': self.loss,
                           'iou': self.iou,
                           }

        torch.save(checkpoint_data, self.path_model_checkpoint)

    def _save_if_best(self):
        # "Best" defined by f1 score on "tumor" class detection
        iou = np.array([x["iou"] for x in self.metrics])
        if iou[-1] == np.max(iou):
            torch.save({'model_state_dict': self.model.state_dict()}, self.best_model_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.path_model_checkpoint)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss = checkpoint["loss"]
        self.metrics = checkpoint["metrics"]

        print(f"Model updated using parameters from: {self.path_model_checkpoint}")

    def train(self, train_loader, validation_loader, n_epochs, verbose=False, ):
        # Loop through epochs
        for epoch in tqdm(range(n_epochs), desc="Epochs"):
            loss_sublist = []

            # Training
            self.model.train()
            for x, y, _ in train_loader:
                # x: image, y: mask, _: label
                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)

                loss = self.criterion(out, y)
                loss_sublist.append(loss.data.item())

                loss.backward()
                self.optimizer.step()

            self.loss.append(np.mean(loss_sublist))
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Model evaluation
            self.model.eval()
            predictions = []
            ground_truths = []

            loss = []
            iou = []

            for x, y, _ in validation_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)

                loss.append(self.criterion(out, y).item())
                iou.append(self.iou(out, y).item())

            iou = np.mean(iou)
            loss = np.mean(loss)

            self.metrics.append({'iou': iou,
                                 'loss': loss})
            self.epoch += 1
            self.save_checkpoint()
            self._save_if_best()

            if verbose:
                print("*" * 25)
                print(f"Summary epoch {self.epoch}:")
                print(f"----Loss train:     \t{self.loss[-1]:.3f} ")
                print(f"----Loss val:       \t{self.metrics[-1]['loss']:.3f}")
                print(f"----IoU val:        \t{self.metrics[-1]['iou']:.3f}")

    def report(self, title=None):
        """
        Make a reporting with three plots :
            1) loss and accuracy VS epoch
            2) "No tumor" label: Recall VS Precision as a function of epoch
            3)  "Tumor"   label: Recall VS Presision as a function of epoch
        """

        loss_trn = self.loss
        loss_val = np.array([x['loss'] for x in self.metrics])
        iou = np.array([x['iou'] for x in self.metrics])

        fig, ax = plt.subplots(1, 2, figsize=(15 / 2.54, 8 / 2.54))
        ax[0].plot(loss_trn, "-o", color='tab:red', label="Train")
        ax[0].plot(loss_val, "-o", color='tab:blue', label="Val")
        ax[0].legend()
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Epoch")

        ax[1].plot(iou, color='tab:blue')
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("IoU")

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.show()


# To avoid data leakage between the train_set and the valid_set between training sessions,
# DO NOT change the seed !
train_set_size = int(len(mri_dataset) * 0.85)
valid_set_size = len(mri_dataset) - train_set_size


train_set, valid_set = random_split(mri_dataset, [train_set_size, valid_set_size],
                                    generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=valid_set, batch_size=1)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SegUNet(dropout=dropout)
model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
# lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=5, mode="triangular2")
lr_scheduler = None
criterion = smp.utils.losses.DiceLoss()


mri_net_trainer = NNTrainer(model, optimizer, criterion, device, lr_scheduler, path_best_model, path_model_checkpoint, True)


mri_net_trainer.train(train_loader, validation_loader, 20, True)


mri_net_trainer.report()


def test_model(model, device, im, mk):
    model.eval()
    im_ = im
    im, mk = im.to(device), mk.to(device)
    im_ = torch.unsqueeze(im, 0)
    pd = torch.squeeze(model(im_), 0)

    print(pd.size())

    im = im.cpu().permute(1, 2, 0).numpy()
    mk = mk.cpu().permute(1, 2, 0).numpy()
    pd = pd.cpu().permute(1, 2, 0).detach().numpy()

    pd = np.round(pd)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im = std * im + mean
    im = np.clip(im, 0, 1)

    #     im = np.where(mk, np.array([[255,0,0]]), im)
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))

    ax[0].imshow(im)
    ax[1].imshow(mk)
    ax[2].imshow(pd)
    ax[0].set_title("MRI")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    plt.axis('off')
    plt.show()


j = 1
for i in range(100):
    im, mk, lb = mri_dataset[i]

    if lb:
        test_model(model, device, im, mk)
        j += 1
    if i >= 5:
        break
