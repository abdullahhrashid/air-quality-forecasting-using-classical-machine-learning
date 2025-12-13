import pandas as pd
import os

#where the file is located
path = os.path.join(os.path.dirname(__file__), '../../data/processed/dataset.csv')

#ingesting dataset
df = pd.read_csv(path)

#needed for using pandas' powerful time series functionality
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

#so that our models knows what day of the year it is since our data is highly seasonal
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['day_of_week'] = df.index.day_of_week
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

#lag features
for i in range(1,8):
    feature = f'lag_{i}'
    df[feature] = df['pm25_level'].shift(i)

#rolling mean for the past 7 days
df['moving_avg_7'] = df['pm25_level'].rolling(window = 7).mean().shift(1)

#rolling std for the past 7 days
df['moving_std_7'] = df['pm25_level'].rolling(window = 7).std().shift(1)

#dropping the first 7 rows due to nans, not a problem since our dataset is quite large
df = df.dropna()

#these columns are useless for pm2.5 levels
df = df.drop(columns = ['feelslikemax', 'feelslikemin', 'feelslike'], axis = 1)

#these two columns are highly correlated to temperature
df = df.drop(columns = ['tempmax', 'tempmin'], axis = 1)

#saving the df
path = os.path.join(os.path.dirname(__file__), '../../data/interim/lagged_dataset.csv')

df.to_csv(path)
