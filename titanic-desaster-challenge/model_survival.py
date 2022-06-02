import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchmetrics
import sklearn.model_selection

from utils import FeatureExtracter, NNTrainer, TitanicDataset
from model_age import load_age_model


def get_survival_model(input_size):
    model = nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          # nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          # nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          # nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(256, 32), nn.ReLU(), nn.Dropout(0.5),
                          nn.Linear(32, 1), nn.Sigmoid(),
                          )
    return model


def load_survival_model(path):
    model = get_survival_model(30)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    path_model_checkpoint = Path("model_survival_checkpoint.pt")
    path_best_model = Path("model_survival_best.pt")

    # Hyperparameter selection
    batch_size = 2          # Dataset
    lr = 0.0001             # optimizer : Adam
    betas = (0.9, 0.999)    # optimizer : Adam
    alpha = 5e-4            # L2 regularization
    epochs = 100            # Algorithm

    # Dataset
    train = pd.read_csv("train.csv", index_col=0)

    train, valid = sklearn.model_selection.train_test_split(train,  train_size=0.75, random_state=42, shuffle=True)
    valid, test = sklearn.model_selection.train_test_split(valid, train_size=0.6, random_state=42, shuffle=False)

    feature_extractor = FeatureExtracter()
    age_model = load_age_model("model_age.pt")

    train_features = feature_extractor.fit_transform(train, age_model)
    valid_features = feature_extractor.transform(valid)
    test_features = feature_extractor.transform(test)

    train_set = TitanicDataset(train_features.values, train["Survived"].values)
    valid_set = TitanicDataset(valid_features.values, valid["Survived"].values)
    test_set = TitanicDataset(test_features.values, test["Survived"].values)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1)
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    x, y = train_set[0]
    print(len(x))

    # Model definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = train_features.shape[1]

    model = get_survival_model(input_size)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    lr_scheduler = None
    criterion = nn.BCELoss()

    def my_accuracy(x, y):
        """Adapter for the metric which requires int as input for label."""
        return torchmetrics.functional.accuracy(x, y.type(torch.int))

    metrics = {"acc": my_accuracy}

    nn_trainer = NNTrainer(model, optimizer, criterion, device, lr_scheduler, alpha, metrics,
                           path_best_model, path_model_checkpoint, True)

    # nn_trainer.train(train_loader, valid_loader, 1, ("acc", "max"), test_loader, verbose=False)
    nn_trainer.report()


if __name__ == "__main__":
    main()
