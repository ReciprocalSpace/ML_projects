import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchmetrics

from sklearn import metrics, preprocessing, model_selection


class FeatureExtracter:
    """Transform the input dataframe into a more "exploitable" one."""
    def __init__(self):
        """Initialize the object properties"""
        self.fitted = False

        self.age_mean = None
        self.age_std = None
        self.fare_mean = None
        self.fare_std = None
        self.bag_of_titles = None
        self.list_of_columns = None
        self.age_model = None

    def fit(self, df: pd.DataFrame, model=None) -> None:
        if self.fitted:
            raise Exception("Object already fitted")

        """Set the object parameters so the every dataset is dealt with the same way."""
        self.fitted = True

        self.age_mean = df["Age"].mean()
        self.age_std = df["Age"].std()
        self.fare_mean = df["Fare"].mean()
        self.fare_std = df["Fare"].std()
        self.bag_of_titles = []
        self.list_of_columns = []

        self.age_model = model

        # We keep only the title from the name and discard the rest
        names = df["Name"]
        df_tmp = pd.DataFrame([name.split(",")[1].split(".")[0].strip() for name in names],
                              columns=["Title"])  # Miss, Mr,...
        value_counts = df_tmp["Title"].value_counts()
        # We keep only the frequent titles, the rare one are replaced by "other", like Jonkheer.
        for k, v in value_counts.items():
            if v > 10:
                self.bag_of_titles.append(k)

        # To select the list_of_columns, we make a first transform of the database. It's not really clean,
        # or efficient, but it works.
        df = self.transform(df)
        self.list_of_columns = list(df.columns)


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted

        # Numerical columns
        out = df[["Age", "Fare"]].copy()

        if self.age_model:
            out["Age"] = (df["Age"] - self.age_mean) / self.age_std
        else:
            out["Age"] = (df["Age"].fillna(self.age_mean) - self.age_mean) / self.age_std

        out["Fare"] = (df["Fare"].fillna(self.fare_mean) - self.fare_mean) / self.fare_std

        # Missing values in cabin
        out["Cabin"] = np.where(df["Cabin"].isnull().values, 0, 1)

        # Conversion of names to titles
        names = df["Name"]
        df_title = pd.DataFrame([name.split(",")[1].split(".")[0].strip() for name in names], columns=["Title"],
                                index=df.index)
        df_ = df_title.copy()
        df_.loc[~df_title["Title"].isin(self.bag_of_titles), "Title"] = "Other"
        df_title = df_

        # One-hot-vector encoding of text-based columns
        out = self.dummization(df_title, ["Title"], out, False)
        out = self.dummization(df, ["Sex", "Pclass", "SibSp", "Parch", "Embarked"], out, False)

        # Making sure that all datasets have the same columns in the same order by:
        # 1) dropping the extra columns
        # 2) adding the missing columns
        # 3) reording the colums alphabetically
        columns = list(out.columns)
        if self.list_of_columns:
            for c in columns:
                if c not in self.list_of_columns:
                    out = out.drop(columns=[c])  # Drop extra columns
            for c in self.list_of_columns:
                if c not in out.columns:
                    out[c] = 0  # Add missing columns

        out = out.reindex(sorted(out.columns), axis=1)  # Reordering

        if self.age_model:
            selection = df['Age'].isnull()
            null_df = out[selection]

            x = torch.tensor(null_df.drop(columns="Age").values).type(torch.float)
            pred = self.age_model(x).cpu().detach().squeeze().numpy()
            out.loc[selection, "Age"] = pred

        return out

    def fit_transform(self, df, age_model=None) -> pd.DataFrame:
        self.fit(df, age_model)
        return self.transform(df)

    @staticmethod
    def dummization(inp, columns, out=None, dummy_na=False):
        if out is None:
            out = pd.DataFrame(index=inp.index)
        for c in columns:
            dummy = pd.get_dummies(inp[c], c, dummy_na=dummy_na)
            out = pd.concat([out, dummy], axis=1)
        return out


class TitanicDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type(torch.float)
        self.y = torch.unsqueeze(torch.from_numpy(y).type(torch.float), dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class NNTrainer:
    #     model, optimizer, criterion, device, lr_scheduler, path_model_checkpoint, False
    def __init__(self, model, optimizer, criterion, device, lr_scheduler=None, alpha=0., metrics=None,
                 path_best_model: Path = None, path_model_checkpoint: Path = None, from_checkpoint=True):
        self.path_model_checkpoint = path_model_checkpoint
        self.best_model_path = path_best_model
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler

        self.alpha = alpha

        self.epoch = 0

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.metrics = metrics if metrics is not None else {}
        self.valid_metrics = {key: [] for key in self.metrics}
        self.test_metrics = {key: [] for key in self.metrics}

        # Check if model already exists
        if from_checkpoint:
            if path_model_checkpoint.is_file():
                self.load_checkpoint()

    def save_checkpoint(self):
        checkpoint_data = {'epoch': self.epoch,
                           'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': self.optimizer.state_dict(),
                           'train_loss': self.train_loss,
                           'valid_loss': self.valid_loss,
                           'test_loss': self.test_loss,
                           'valid_metrics': self.valid_metrics,
                           'test_metrics': self.test_metrics,
                           }

        torch.save(checkpoint_data, self.path_model_checkpoint)

    def load_checkpoint(self):
        checkpoint = torch.load(self.path_model_checkpoint)

        self.epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.valid_loss = checkpoint["valid_loss"]
        self.test_loss = checkpoint["test_loss"]
        self.valid_metrics = checkpoint["valid_metrics"]
        self.test_metrics = checkpoint["test_metrics"]

        print(f"Model loaded using parameters from: {self.path_model_checkpoint}")

    def _save_if_best(self, best_metric):
        metric, mode = best_metric
        # "Best" defined by f1 score on "tumor" class detection
        score = self.valid_metrics[metric]
        if mode == "max" and score[-1] == np.max(score):
            torch.save({'model_state_dict': self.model.state_dict()}, self.best_model_path)
            print(f"Best model epoch: Epoch {self.epoch}")
        elif mode == "min" and score[-1] == np.min(score):
            torch.save({'model_state_dict': self.model.state_dict()}, self.best_model_path)
            print(f"Best model epoch: Epoch {self.epoch}")

    def _train_epoch(self, data_loader):
        # Training
        self.model.train()
        epoch_loss = []
        for x, y in data_loader:
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            z = self.model(x)

            loss = self.criterion(z, y)
            l2_loss = 0
            if self.alpha != 0:
                for p in self.model.parameters():
                    l2_loss += self.alpha * torch.square(p).sum()
            loss.backward()
            self.optimizer.step()

            epoch_loss.append(loss.data.item())

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return np.mean(epoch_loss)

    def _eval_epoch(self, valid_loader):
        self.model.eval()
        epoch_loss = []
        epoch_metrics = {}

        scores = [[] for _ in self.metrics]
        for x, y in valid_loader:
            y_int = y.type(torch.int)
            y_int = y_int.to(self.device)

            x, y = x.to(self.device), y.to(self.device)
            z = self.model(x)

            loss = self.criterion(z, y)
            epoch_loss.append(loss.data.item())
            for score, metric in zip(scores, self.metrics.values()):
                score.append(metric(z, y).cpu().detach().numpy())

        return np.mean(epoch_loss), np.mean(scores, axis=1)

    def train(self, train_loader, valid_loader, n_epochs, best_metric, test_loader=None, verbose=False, ):
        # Loop through epochs
        for epoch in tqdm(range(n_epochs), desc="Epochs"):
            train_loss = self._train_epoch(train_loader)
            valid_loss, valid_scores = self._eval_epoch(valid_loader)

            if test_loader:
                test_loss, test_scores = self._eval_epoch(test_loader)
                self.test_loss.append(test_loss)
                for test_metric, score in zip(self.test_metrics.values(), test_scores):
                    test_metric.append(score)

            # Savin results
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            for valid_metric, score in zip(self.valid_metrics.values(), valid_scores):
                valid_metric.append(score)

            self.epoch += 1
            self.save_checkpoint()
            self._save_if_best(best_metric)

            if verbose:
                print("*" * 25)
                print(f"Summary epoch {self.epoch}:")
                print(f"----Loss train:     \t{self.loss[-1]:.3f} ")
                print(f"----Loss val:       \t{self.metrics[-1]['loss']:.3f}")
                print(f"----IoU val:        \t{self.metrics[-1]['iou']:.3f}")

    def report(self, title=None):

        n = len(self.valid_metrics)+1
        fig, ax = plt.subplots(1, n, figsize=( (5*n+3) / 2.54, 8 / 2.54))
        ax = ax if n > 1 else [ax]

        ax[0].plot(self.train_loss, "-o", color='tab:red', label="Train")
        ax[0].plot(self.valid_loss, "-o", color='tab:blue', label="Valid")
        if self.test_loss:
            ax[0].plot(self.test_loss, "-+", color='tab:green', label="Test")
        ax[0].legend()
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Epoch")
        for i, (name, metric) in enumerate(self.valid_metrics.items()):
            j = i+1
            ax[j].plot(metric, "-o", color='tab:blue', label="Valid")
            ax[j].legend()
            ax[j].set_ylabel(name)
            ax[j].set_xlabel("Epoch")
            if self.test_loss:
                ax[j].plot(self.test_metrics[name], "-+", color='tab:green', label="Test")

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.show()