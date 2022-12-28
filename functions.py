from pymongo import MongoClient
import pandas as pd

def pre_processer(df):
  """
  Função criada para remover os dados nulos e faltantes antes do 
  treinamento do modelo.

  df: DataFrame montado a partir dos dados 'brutos' do MongoDB
  return: DataFrame sem zeros e valores faltantes.
  """
  #Eliminando valores duplicados
  if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

  #Eliminando registros nulos
  if df.isna().sum().sum() > 0:
    df = df.dropna(how='any')

  #Eliminando registros com preço = 0
  if (df['price'] == 0).sum() > 0:
    df = df.drop(
        index=df[(df['price'] == 0)]\
        .index
    )
  return df


def matrix_mounter(neighbors, df_processed):
    """
    Função para montagem da matriz de semelhança usando os 
    índicies dos neighbors retornados pelo NearestNeighbors

    neighbors:  Array com os índices
    df_processed: Dataframe usado para o treinamento do modelo
    return: Matriz de similaridade processada com os ID's dos imóveis
    """
    s_matrix = pd.DataFrame(neighbors)\
    .apply(lambda x: df_processed.index[x])\
    .rename(columns={
        0: 'unit_id',
        1: 'neighbor_1',
        2: 'neighbor_2',
        3: 'neighbor_3',
        4: 'neighbor_4',
        5: 'neighbor_5',
        6: 'neighbor_6',
        7: 'neighbor_7',
    })\
    .set_index('unit_id')

    return s_matrix
