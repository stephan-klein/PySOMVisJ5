import panel as pn
import numpy as np
import pandas as pd

loaded_weights = np.loadtxt('datasets/room_occupancy/som_occ_s.csv', delimiter=',')

# Reshape the array to its original dimensions (60, 40, 18)
weights = loaded_weights.reshape((60, 40, 18))


# Path to your CSV file
file_path = 'datasets/room_occupancy/Occupancy_Estimation.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])
# Find the earliest date
earliest_date = df['Date'].min()
# Calculate the difference in days from the earliest date
df['Days_Since_Earliest'] = (df['Date'] - earliest_date).dt.days

df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
# Calculate the number of seconds since the beginning of the day
df['Seconds_Since_Midnight'] = df['Time'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

df.drop(['Date'], axis=1, inplace=True)
df.drop(['Time'], axis=1, inplace=True)
df.drop(['Room_Occupancy_Count'], axis=1, inplace=True)

columns_to_normalize = df.select_dtypes(include=np.number).columns
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()

from pysomvis import PySOMVis
vis = PySOMVis(weights=weights, input_data=df.values)
pn.serve(vis._mainview, port=44141, show=True)