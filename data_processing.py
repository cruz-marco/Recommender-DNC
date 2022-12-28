#Preprocessamento de dados
from database import data_getter
from functions import pre_processer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Definição das variáveis de interesse
FEATURES = ['bathrooms', 'bedrooms', 'living_area', 'latitude',
            'longitude', 'price', 'property_type_id']

#Montagem do dataframe principal
df = pd.DataFrame(
    data = data_getter()    
    ).set_index(
        'unit_id'
    )[FEATURES] 

df = pre_processer(df)

#Aplicando o log em living_area e em price
df[['living_area', 'price']] = np.log(df[['living_area', 'price']])

#Aplicando o MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df)
scaled_feats = scaler.transform(df)

df[FEATURES] = scaled_feats
