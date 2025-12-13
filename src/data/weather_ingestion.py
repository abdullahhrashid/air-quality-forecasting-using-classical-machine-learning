import requests
import pandas as pd
import os

#params for the api call
API_KEY = 'removed due to obvious reasons'
location = 'new%20delhi'
start = '2024-12-01'
end   = '2025-12-01'

#url for the api call
url = (
    f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    f'{location}/{start}/{end}'
    f'?unitGroup=metric&include=days&key={API_KEY}&contentType=json'
)

#extracting the json response
resp = requests.get(url)
data = resp.json()

#needed for parsing the json response correctly
df = pd.json_normalize(data, record_path=['days'])

#needed due to python's shenanigans
path = os.path.join(os.path.dirname(__file__), '../../data/raw/meteorological/weather2.csv')

#saving the data to a csv file
df.to_csv(path, index=False)