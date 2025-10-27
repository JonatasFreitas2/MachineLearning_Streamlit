import streamlit as st
import pandas as pd
import joblib
import os

#nome_usuario = st.text_input("Informe seu nome", placeholder="Digite aqui...")


#if st.button(label="Clique aqui: "):
    #st.write("Seja bem vindo, " + nome_usuario)

#ETAPA 01 - DEFINIÇÃO DAS FEATURES
FEATURES_NAMES = [
    'Nota_P1',
    'Nota_P2',
    'Media_Trabalhos',
    'Frequencia',
    'Reprovacoes_Anteriores',
    'Acessos_Plataforma_Mes'
]

COLUNAS_HISTORICO = FEATURES_NAMES + ["Previsao_Resultado", "Prob_Aprovado", "Prob_Reprovado"]

#Criar uma sessão st.session
if 'previsoes' not in st.session_state:
    st.session_state.previsoes = pd.DataFrame(columns=COLUNAS_HISTORICO)

#ETAPA 02 - CARREGAMENTO DO MODELO PARA NOSSO FRONT-END
#st.cache_resource para carregar o modelo apenas uma vez
#Otimizando o desempenho do app
@st.cache_resource
def carregar_modelo(caminho_modelo = "modelo_previsao_desempenho.joblib"):
    """
        Carregar o pipeline de ML treinado (scaler + modelo) do arquivo .joblib
    """

    try:
        if os.path.exists(caminho_modelo):
            modelo = joblib.load(caminho_modelo)
            return modelo
        else:
            st.error(f"Erro: Arquivo do modelo '{caminho_modelo}' não foi encontrado!")
            st.warning(f'Por favor, execute o script "modelo_treinamento.py" para gerar o modelo')
        
    except Exception as e:
        st.error(f"Erro inesperado ao carregar o modelo: {e}")
        return None
    
pipeline_modelo = carregar_modelo()

#ETAPA 02: CONFIGURAÇÃO DA INTERFACE DO USUARIO(STREAMLIT)

st.set_page_config(layout='wide', page_title="Previsao de notas")

st.title("🤖 Previsor de desempenho acadêmico")
st.markdown("""
    Essa ferramenta usa Inteligencia Artificial para prever os status final (Aprovado ou reprovado) de um aluno com base em seu desempenho parcial.
            
    **Preencha os dados do aluno abaixo para obter uma previsão: **

""")

#ETAPA 03 - FORMULÁRIO DE ENTRADA
if pipeline_modelo is not None:
    #Utilizar um formulario para agrupar as entradas e o botão
    with st.form("Formulario_previsao"):
        st.subheader("Insira as notas e métricas do aluno")

        col1, col2 = st.columns(2)

        with col1:
            nota_p1 = st.slider("Nota da P1 (0 a 10)", min_value=0.0, max_value=10.0, value= 5.0, step=0.5)
            media_trabalhos = st.slider("Media dos trabalhos (0 a 10)", min_value=0.0, max_value=10.0, value= 5.0, step=0.5)
            reprovacoes_ant = st.number_input("Reprovações anteriores", min_value=0, max_value=10, value=0, step=1)

        with col2:
            nota_p2 = st.slider("Nota da P2 (0 a 10)", min_value=0.0, max_value=10.0, value= 5.0, step=0.5)
            frequencia = st.slider("Frequencia (%)", min_value=0, max_value=100, value=75, step=5)
            acesso_mes = st.number_input("Média de acessos a plataforma (por mês)", min_value=0, max_value=100, value=10, step=1)


        submitted = st.form_submit_button("Realizar previsão")

    if submitted:

        features_name = [
            'ï»¿Nota_P1',
            'Nota_P2',
            'Media_Trabalhos',
            'Frequencia',
            'Reprovacoes_Anteriores',
            'Acessos_Plataforma_Mes'
        ]

        #Criação de um dataFrame a partir dos dados inseridos
        dados_alunos = pd.DataFrame(
            [[nota_p1, nota_p2, media_trabalhos, frequencia, reprovacoes_ant, acesso_mes]],
            columns=features_name
        )
        
        st.info("Processando dados e realizando a previsão...")

        try:
            #Realizar a previsão([0] ou [1])
            previsao = pipeline_modelo.predict(dados_alunos)

            #Obter a probabilidade 
            probabilidade = pipeline_modelo.predict_proba(dados_alunos)

            prob_reprovados = probabilidade[0][0]
            prob_aprovados = probabilidade[0][1]
            resultado_texto = "APROVADO" if previsao[0] == 1 else "REPROVADO"

            #Exibir os resultados na tela
            st.subheader("Resultado da previsão")
            
            if previsao[0] == 1:
                st.success("Previsão: aprovado!")
                st.markdown(f"""
                Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                **{prob_aprovados*100:.2f}%** de chance de ser **aprovado**

                *Chance de reprovação: {prob_reprovados*100:.2f}%*
            """)
                

            else:
                st.success("Previsão: reprovado (zona de risco)")
                st.markdown(f"""
                Com base nos dados fornecidos, o modelo prevê que o aluno tem:
                **{prob_reprovados*100:.2f}%** de chance de ser **reprovado**

                *Chance de aprovação: {prob_aprovados*100:.2f}%* 
            """)
                
            nova_linha_dict = {
                'Nota_P1': nota_p1,
                'Nota_P2': nota_p2,
                'Media_Trabalho': media_trabalhos,
                'Frequencia': frequencia,
                'Reprovacoes_Anteriores': reprovacoes_ant,
                'Acessos_Plataforma_Mes': acesso_mes,
                'Previsao_Resultado': resultado_texto,
                'Prob_Aprovado': prob_aprovados,
                'Prob_Reprovado': prob_reprovados
            }

            nova_linha_df = pd.DataFrame([nova_linha_dict], columns=COLUNAS_HISTORICO)

            st.session_state.previsoes = pd.concat(
                [st.session_state.previsoes, nova_linha_df],
                ignore_index=True
            )

        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {e}")
            st.error("Verifique se os nomes das colunas correspondem aos nomes das colunas utilizadas nos treinos")

    st.subheader("Histórico de previsões realizadas na sessão: ")

    if st.session_state.previsoes.empty:
        st.write("Nenhuma previsão foi realizada ainda")
    else:
        st.dataframe(st.session_state.previsoes, use_container_width=True)

        if st.button("Limpar histórico"):
            st.session_state.previsoes = pd.DataFrame(columns= COLUNAS_HISTORICO)

            st.rerun()

else:
    st.warning("O aplicativo não pode fazer previsões porque o modelo não foi carregado!")