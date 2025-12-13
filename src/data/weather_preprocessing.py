import pandas as pd
import os

#i love python
path1 = os.path.join(os.path.dirname(__file__), f'../../data/raw/meteorological/weather2.csv')
path2 = os.path.join(os.path.dirname(__file__), f'../../data/raw/meteorological/weather.csv')

#reading csv files
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

df1['datetime'] = pd.to_datetime(df1['datetime'])
df2['DATE'] = pd.to_datetime(df2['DATE'])

#extracting only rows of years after 2018
df2 = df2[df2['DATE'].dt.year > 2018]

#removing the columns that aren't in the second csv
df1 = df1.drop(columns = ['datetimeEpoch', 'dew', 'preciptype', 'snow', 'snowdepth', 'windgust',
       'winddir', 'cloudcover', 'visibility', 'solarradiation', 'solarenergy', 'uvindex',
       'sunrise', 'sunriseEpoch', 'sunset', 'sunsetEpoch', 'moonphase','description', 
       'icon', 'stations', 'source'])

#removing the columns that aren't in the first csv
df2 = df2.drop(columns = ['Unnamed: 0', 'Year', 'month', 'dayofweek', 'dayofyear', 
       'year-2000', 'weekofyear', 'tempmax_humidity', 'tempmin_humidity', 'temp_humidity',
       'feelslikemax_humidity', 'feelslikemin_humidity', 'feelslike_humidity', 'temp_range',
       'heat_index'])


#renaming the columns in df2 so that they align with df1
df2 = df2.rename(columns = {'DATE' : 'datetime', 'sealevelpressure' : 'pressure'})

#concatenating both csv files into one consolidated csv
df = pd.concat([df2, df1])

#needed for later on
df = df.sort_values('datetime').set_index('datetime')

#saving csv
path = os.path.join(os.path.dirname(__file__), '../../data/processed/weather.csv')
df.to_csv(path)
