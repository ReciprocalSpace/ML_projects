import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchmetrics

from utils import FeatureExtracter, NNTrainer, TitanicDataset
import sklearn.model_selection


def load_age_model(path):
    model = get_model(29)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def get_model(input_size):
    model = nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 32), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(32, 1),
                          )
    return model


def main():
    path_model_checkpoint = Path("model_age_checkpoint.pt")
    path_best_model = Path("model_age.pt")

    # Hyperparameter selection
    batch_size = 5  # Dataset
    lr = 0.0001  # optimizer : Adam
    betas = (0.9, 0.999)  # optimizer : Adam
    alpha = 5e-4  # L2 regularization
    epochs = 200  # Algorithm

    # DATASET
    df = pd.read_csv("train.csv", index_col=0)
    fit_df, _ = sklearn.model_selection.train_test_split(df, train_size=0.75, random_state=42, shuffle=True)
    df_age = df[~df['Age'].isnull()]

    feature_extractor = FeatureExtracter()
    feature_extractor.fit(fit_df)
    age_set = feature_extractor.transform(df_age)

    train_features, valid_features = sklearn.model_selection.train_test_split(
        age_set, train_size=0.80, random_state=42, shuffle=True)

    train_set = TitanicDataset(train_features.drop(columns="Age").values, train_features["Age"].values)
    valid_set = TitanicDataset(valid_features.drop(columns="Age").values, valid_features["Age"].values)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1)

    # Neural Network
    x, y = train_set[0]
    input_size = len(x)
    print(input_size)
    model = get_model(input_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    lr_scheduler = None
    criterion = nn.MSELoss()

    metrics = {"loss": nn.MSELoss()}

    nn_trainer = NNTrainer(model, optimizer, criterion, device, lr_scheduler, alpha, metrics, path_best_model,
                           path_model_checkpoint, False)
    nn_trainer.train(train_loader, valid_loader, epochs, ("loss", "min"), verbose=False)

    nn_trainer.report("")


if __name__ == "__main__":
    main()
