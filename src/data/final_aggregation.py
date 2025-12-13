import pandas as pd
import glob
import os

#path containing csv files
path = os.path.join(os.path.dirname(__file__), '../../data/processed')
csv_path = os.path.join(path, '*.csv')

#extracting csv file paths for reading
csv_list = glob.glob(csv_path)

#list to hold dfs
df_list = []

#reading dfs
for csv in csv_list:
    df_list.append(pd.read_csv(csv))

#some changes needed for merging
df_list[0] = df_list[0].rename(columns = {'value' : 'pm25_level'})
df_list[0]['datetime'] = pd.to_datetime(df_list[0]['datetime']).dt.date
df_list[1] = df_list[1].rename(columns = {'valid_time' : 'datetime'})

final_list = []

for df in df_list:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.drop_duplicates(subset='datetime')
    df = df.set_index('datetime')
    final_list.append(df)

#merging the csv
df = pd.concat(final_list, axis = 1, join = 'inner')

df = df.sort_index()

#path for saving the dataset
df_path = os.path.join(os.path.dirname(__file__), '../../data/processed/dataset.csv')

#saving csv
df.to_csv(df_path)