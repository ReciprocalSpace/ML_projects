import numpy as np
from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier


path = Path("../datasets/movie.csv")
df = pd.read_csv(path, delimiter=",")

print(df.head())

X = df["text"].values
y = df["label"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
pre_pipe = Pipeline(steps=[("to-bag-of-words", CountVectorizer()),
                           ("normalize", TfidfTransformer())])

X_train = pre_pipe.fit_transform(X_train)
X_test = pre_pipe.transform(X_test)


print(df.head())

vec = CountVectorizer()

TfidfTransformer(min_df )