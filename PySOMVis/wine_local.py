import panel as pn
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

loaded_weights = np.loadtxt('datasets/wine_quality/som_wines_s.csv', delimiter=',')

# Reshape the array to its original dimensions (60, 40, 18)
weights = loaded_weights.reshape((60, 40, 12))


# Path to your CSV file
file_path = 'datasets/wine_quality/som_wines_s.csv'
df = pd.read_csv(file_path)


# fetch dataset
wine_quality = fetch_ucirepo(id=186)

# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

# metadata
print(wine_quality.metadata)

# variable information
print(wine_quality.variables)

dataset = wine_quality.data.original

df = dataset
# Create classes from quality column
quality_classes = y.to_numpy().flatten()
quality_classes_str = quality_classes.astype(str)

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