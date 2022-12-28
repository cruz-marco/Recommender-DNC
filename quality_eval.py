from database import data_getter
from data_processing import FEATURES
import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.6f}'.format


def matrix_evaluator(sim_matrix, dists):
  #Buscando propriedades dos imóveis
  properties = pd.DataFrame(
                data = data_getter()    
                ).set_index(
                    'unit_id'
                )[FEATURES]\
                  .drop_duplicates()

  p_cols = list(properties.reset_index().columns) #Salvando a lista com nomes das colunas

  SAMPLE_SIZE = 15 #quantidade de registros para avaliação da amostra.

  #Buscando as distâncias para avaliação
  dists = pd.DataFrame(dists, index=sim_matrix.index)\
          .drop(columns=[0])\
          .rename(columns={
          1: 'neighbor_1',
          2: 'neighbor_2',
          3: 'neighbor_3',
          4: 'neighbor_4',
          5: 'neighbor_5',
          6: 'neighbor_6',
          7: 'neighbor_7'
          })

  #Gerando uma amostra aleatória
  samples = list(np.random.choice(sim_matrix.index, size=SAMPLE_SIZE))

  eval_df = pd.DataFrame()# Dataframe principal a ser retornado

  #Gerando uma linha em branco no dataframe
  blanked = pd.Series({k:v for (k,v) in zip(p_cols, 
                      ['---' for _ in range(len(p_cols))])})\
                      .to_frame().T
  blanked['distancies'] = '---'

  titles = pd.Series({k:v for (k,v) in zip(p_cols, p_cols)}).to_frame().T
  titles['distancies'] = 'distancies'

  #Loop para buscar as informações dos imóveis, cruzando com os IDs da matriz de similaridade.
  for sample in samples:
    #Busca as infos do imóvel de referência.
    ref = pd.DataFrame(properties.loc[sample]).T\
      .reset_index()\
      .rename(columns={'index': 'unit_id'})
    ref['distancies'] = 'reference'

    #Basca as infos das recomendações
    recs = properties.loc[sim_matrix.loc[sample].to_list()].reset_index()
    recs['distancies'] = dists.T[int(sample)].reset_index(drop=True)
    
    #Junta tudo neste DF
    joined = pd.concat([ref, recs]).reset_index(drop=True)

    #Acumula no DF principal de avaliação
    eval_df = pd.concat([eval_df, joined, blanked, titles])

  eval_df = eval_df.reset_index(drop=True)
  eval_df.drop(index=list(eval_df.iloc[-2:].index), inplace=True)

  return eval_df
  
