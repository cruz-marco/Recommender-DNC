from pymongo import MongoClient


#String de conexão com o mongodb
CONNECT_STRING = None

#Nome da base de dados
DB_NAME = None #Nome da coleção que será consumida pelo treinamento do modelo
INPUT_NAME = 'data_portal' #Nome da coleção onde estão os dados de entrada.
OUTPUT_NAME = None #Nome da coleção onde será armazenada a matriz de similaridade.

client = MongoClient(CONNECT_STRING) #Criando o cliente de acesso
db = client[DB_NAME]#Selecionando o banco de dados

# Função para receber os dados em formato de lista.
def data_getter():
    return list(db[INPUT_NAME].find())

# Função para salvar a matriz no banco de dados.
def matrix_saver(matrix):
    return db[OUTPUT_NAME].insert_many(matrix.to_dict('records'))
