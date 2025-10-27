import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics


#ETAPA 01 - CARREGAR DADOS
def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:
        #carregamento de dados
        if os.path.exists(caminho_arquivo):

            df= pd.read_csv(caminho_arquivo, encoding="latin1", sep=',')

            print("O arquivo foi carregado com sucesso!")

            return df
        else:
            print("O arquivo não foi encontrado dentro da pasta!")

            return None

    except Exception as e:
        print("Erro inesperado ao carregar o arquivo: ", e)

        return None
    
# ---- chamar a função para armazenar o resultado ----

dados = carregar_dados()


#-------- ETAPA 02 : PREPARAÇÃO E DIVISÃO DE DADOS ----------
# Definição de x (Features) e y (Target)

if dados is not None:
    print(f"\nTotal de Registros Carregados: {len(dados)}")
    print("Iniciando o pipeline de treinamento")

    target_column = "Status_Final"

#ETAPA 2.1 - DEFINIÇÃO DAS FEATURES E TARGET
    try:
        X = dados.drop(target_column, axis=1)
        y = dados[target_column]
        print(f"Features (x) definidas: {list(X.columns)}")
        print(f"Features (y) definidas {target_column}")

    except KeyError:
        
        print(f"\n ----- Erro Critico -------")
        print(f"A coluna {target_column} não foi encontradado no CSV")
        print(f"Colunas disponiveis: {list(dados.columns)}")
        print(f"Por favor, ajuste a variavel 'target_column' e tente novamente!")
        #Se o target não for encontrado, irá encerrar o script!
        exit()

#ETAPA 2.2 - DEVISÃO ENTRE TREINO E TESTE
    print("\n ------- Dividindo dados em treino e teste... ------")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=0.2,     #20% dos dados serão utilizados para teste
        random_state=42,   #Garantir a reprodutibilidade
        stratify=y         #Manter a proporção de aprovados e reprovados

    )

    print(f"Dados de treino: {len(X_train)} | Dados de teste: {len(X_test)}")

    #ETAPA 03: CRIAÇÃO DA PIPELINE DE ML

    print("\n ------- Criando a pipeline de ML... --------")
    #scaler -> normalização dos dados (colocando tudo na mesma escala)
    #model -> aplica o modelo de regreção logistica
    pipeline_model = pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])

    #ETAPA 04: TREINAMENTO E AVALIAÇÃO DOS DADOS/MODELO
    print("\n ------ Treinamento do modelo... --------")
    #Treina a pipeline com os dados de treino
    pipeline_model.fit(X_train, y_train)

    print("Modelo treinado. Avaliando com os dados de teste...")
    y_pred = pipeline_model.predict(X_test)

    #AVALIAÇÃO DE DESEMPENHO
    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)

    print("\n -------- Relatório de avaliação geral ---------")
    print(f"Acurácia Geral: {accuracy * 100:.2f}%")
    print("\n Relatório de classificação detalhado")
    print(report)


    #ETAPA 05: SALVANDO O MODELO
    model_filename = 'modelo_previsao_desempenho.joblib'

    print(f"\n Salvando o pipeline treinado em... {model_filename}")
    joblib.dump(pipeline_model, model_filename)

    print("Processo concluido com sucesso!")
    print(f"O arquivo '{model_filename}' está para ser utilizado! ")

else:
    print("O pipeline não pode continuar pois os dados não foram carregados!")
