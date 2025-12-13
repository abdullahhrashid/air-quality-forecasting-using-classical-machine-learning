import pandas as pd
import numpy as np
import os

#path to dataset
path = os.path.join(os.path.dirname(__file__), '../../data/interim/2lagged_dataset.csv')

df = pd.read_csv(path)

df['datetime'] = pd.to_datetime(df['datetime'])

#one hot encoding the condtions column
df = pd.get_dummies(df, columns = ['conditions'], drop_first = False, dtype = int)

#cyclical encoding for month
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)

#cyclical encoding for day
df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
df['day_sin'] = np.sin(2 * np.pi * df['day']/31)

#splitting the data
train_set = df[df['datetime'].dt.year != 2025].copy()
test_set = df[df['datetime'].dt.year == 2025].copy()

#not needed anymore
train_set = train_set.drop(columns = ['year', 'month', 'day', 'datetime'])
test_set = test_set.drop(columns = ['year'])

train_path = os.path.join(os.path.dirname(__file__), '../../data/train_set.csv')
test_path = os.path.join(os.path.dirname(__file__), '../../data/test_set.csv')

#saving
train_set.to_csv(train_path, index = False)
test_set.to_csv(test_path, index = False)
