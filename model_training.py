#Treinamento do modelo
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from data_processing import df
from functions import matrix_mounter
from database import matrix_saver
from quality_eval import matrix_evaluator
import datetime
#import mlflow

run_name = f'KNN_Train_{str(datetime.datetime.now())}'
#mlflow.set_experiment(run_name)

#mlflow.start_run()

model = NearestNeighbors(n_neighbors=8, metric='cosine', algorithm='brute', 
                        n_jobs=-1)

#mlflow.log_params(model.get_params())


model.fit(df.to_numpy())

#Executando a predição. Retornando as distâncias e os índices dos vizinhos.
distances, neighbors = model.kneighbors(X=df.to_numpy())

#Monta a matriz de similaridade em um DataFrame, com os índices substituídos
#pelos 'unity_id' dos imóveis.
similarity_matrix = matrix_mounter(neighbors, df)

#Salva um dataframe em html na pasta test_logs para avaliação qualitativa do modelo. Pode ser desabilitada caso necessário.
matrix_evaluator(similarity_matrix, distances).to_html(f'./test_logs/{run_name}.html') 
#mlflow.log_artifact(f'./test_logs/{run_name}.html')


#matrix_saver(similarity_matrix) #Retirar este comentário somente depois de verificar todo o código referente ao banco de dados.

#mlflow.end_run()
