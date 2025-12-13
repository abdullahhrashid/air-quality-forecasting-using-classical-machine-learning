import pandas as pd
import numpy as np
import os 

#again, this path nonsense is needed unfortunately due to how python works
file_path = '../../data/interim/pm25data.csv'

file_path = os.path.join(os.path.dirname(__file__), file_path)

df = pd.read_csv(file_path)

#dropping irrelevant features
df = df.drop(['Unnamed: 0', 'location_id', 'sensors_id', 'location',
       'lat', 'lon', 'parameter', 'units'], axis = 1)

#replacing the negative values
df['value'] = df['value'].apply(lambda x: np.nan if x < 0 else x)

#replacing the strange 1985 value
df['value'] = df['value'].replace(1985, np.nan)

#converting the date to the proper format
df['datetime'] = pd.to_datetime(df['datetime'])

#needed for resampling
df = df.sort_values('datetime').set_index('datetime')

#this gives us the average value over all hours for each day
df = df.resample('D').mean()

path = os.path.join(os.path.dirname(__file__),'../../data/interim/pm25_for_eda.csv')

#so that we can check out the data finally
df.to_csv(path)

#after eda, it seems there are a lot of missing values that we need to account for
#they are small consuective runs of missing values and very large ones
#since the data is very seasonal and shows the same pattern every year, imputing with the
#mean of other years seems to me to be a good approach

#gets the average value for each day of the year over all years
average_values = df.groupby(df.index.dayofyear)['value'].transform('mean')

#imputing nulls
df['value'] = df['value'].fillna(average_values)

path = os.path.join(os.path.dirname(__file__),'../../data/processed/pm25.csv')

#final csv with daily data and missing values handled 
df.to_csv(path)
