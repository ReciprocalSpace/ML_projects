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
from model_survival import load_survival_model

from utils import FeatureExtracter


def main():
    model_age = load_age_model("model_age.pt")
    model_survival = load_survival_model("model_survival_best.pt")

    feature_extractor = FeatureExtracter()
    train = pd.read_csv("train.csv")
    train, _ = sklearn.model_selection.train_test_split(train, train_size=0.75, random_state=42, shuffle=True)
    age_model = load_age_model("model_age.pt")
    feature_extractor.fit(train, age_model)

    test = pd.read_csv("test.csv", index_col=0)
    test_features = feature_extractor.transform(test)
    test_set = TitanicDataset(test_features.values, np.zeros(len(test_features)))

    x_test = test_set.x
    model_survival.eval()
    z = model_survival(x_test)
    z = torch.round(z).detach().numpy().astype(int)

    to_save = pd.DataFrame(z, index=test.index, columns=["Survived"])

    to_save.to_csv("Submission.csv")

if __name__ == "__main__":
    main()