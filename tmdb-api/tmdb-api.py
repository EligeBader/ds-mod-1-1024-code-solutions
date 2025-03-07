# -*- coding: utf-8 -*-
"""tmdb-api.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WOYvM_AYn0Ati3z5FMauKssWOFl5ZoG3
"""

import json
import requests
import pandas as pd

api_key ="7db3815fb7015da6cb2afd43b74d055e"

data_json_list = []
total_pages = 1
url = f"https://api.themoviedb.org/3/search/movie?query=Star%20Wars&api_key={api_key}&page=1"
data = requests.get(url)
data_json = json.loads(data.content)

for i in range(1, data_json['total_pages']+1):
    url = f"https://api.themoviedb.org/3/search/movie?query=Star%20Wars&api_key={api_key}&page={i}"
    data = requests.get(url)
    data_json = json.loads(data.content)

    for movie in data_json['results']:
        data_json_list.append(movie)

data_json_list

data_json.keys()

df = pd.DataFrame(data_json_list)
df

df['popularity']

df_sorted = df.sort_values(by='popularity', ascending=False)
df_sorted

