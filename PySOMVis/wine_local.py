import panel as pn
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

loaded_weights = np.loadtxt('PySOMVis\\datasets\\wine_quality\\som_wines_s.csv', delimiter=',')

# Reshape the array to its original dimensions (60, 40, 18)
weights = loaded_weights.reshape((60, 40, 12))

# fetch dataset
wine_quality = fetch_ucirepo(id=186)

dataset = wine_quality.data.original
sorted_indices = dataset['quality'].argsort()
sorted_dataset = dataset.iloc[sorted_indices]

df = sorted_dataset
# Create classes from quality column
quality_classes = df['quality']
quality_classes_str = np.unique(quality_classes).astype(str)

# Select columns to normalize
df.drop(['quality'], axis=1, inplace=True)
columns_to_normalize = df.select_dtypes(include=np.number).columns

# Z-score normalization
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()

# Normalize wine color
df['color'] = df['color'].map({'white': 0, 'red': 1})

from pysomvis import PySOMVis
vis = PySOMVis(weights=weights, input_data=df.values,classes_names=quality_classes_str,classes=quality_classes)
pn.serve(vis._mainview, port=44141, show=True)