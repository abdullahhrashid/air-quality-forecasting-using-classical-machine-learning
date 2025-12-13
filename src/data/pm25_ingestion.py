import pandas as pd
import os
import glob

#a list of all the years whose data we have
years = ['2019', '2020', '2021', '2022', '2023', '2024', '2025']

data = []

path = '../../data/raw/pm25'

#this path nonsense is needed unfortunately due to how python works
path = os.path.join(os.path.dirname(__file__), path)

#loop to extract data from the years 
for year in years:

    for month in sorted(os.listdir(os.path.join(path, year))):
    
        month_path = f'{path}/{year}/{month}'

        files = glob.glob(f'{month_path}/*.gz')

        for file in files:

            data.append(pd.read_csv(file, compression = 'gzip'))

#concatenating the dfs to a single df
data = pd.concat(data, ignore_index = True)

path = '../../data/interim/'

path = os.path.join(os.path.dirname(__file__), path)

#converting df to csv
df = data.to_csv(os.path.join(path, 'pm25data.csv'), index = 'false')
