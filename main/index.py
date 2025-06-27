import streamlit as st
import pandas as pd
import joblib
import numpy as np
from processamento import preprocessar_texto
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(
    page_title="Classificador de Dificuldades Acadêmicas",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar modelos e recursos
@st.cache_resource
def carregar_modelos():
    try:
        modelo = joblib.load(r'C:\Users\victo\OneDrive\Área de Trabalho\trabalho_topicos\modelo_naive_bayes.pkl')
        vectorizer = joblib.load(r'C:\Users\victo\OneDrive\Área de Trabalho\trabalho_topicos\vectorizer.pkl')
        categorias = {
            0: 'Dificuldade com Leitura',
            1: 'Dificuldade com Matemática',
            2: 'Ansiedade/Estresse',
            3: 'Falta de Concentração',
            4: 'Problemas de Organização',
            5: 'Dificuldade com Prazos'
        }
        return modelo, vectorizer, categorias
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None, None, None

modelo, vectorizer, categorias = carregar_modelos()

# Interface do usuário
def main():
    st.title("📚 Classificador de Dificuldades Acadêmicas")
    st.markdown("""
    Este sistema utiliza Processamento de Linguagem Natural (PLN) para classificar relatos de estudantes 
    em categorias de dificuldades acadêmicas. Insira um texto no campo abaixo para análise.
    """)
    
    # Sidebar com informações
    with st.sidebar:
        st.header("Sobre o Projeto")
        st.markdown("""
        **Objetivo**: Classificar textos de estudantes em categorias de dificuldades acadêmicas para auxiliar professores e gestores.
        
        **Tecnologias**:
        - Python
        - Streamlit (interface)
        - Scikit-learn (modelos de ML)
        - NLP (pré-processamento)
        
        **Categorias**:
        1. Dificuldade com Leitura
        2. Dificuldade com Matemática
        3. Ansiedade/Estresse
        4. Falta de Concentração
        5. Problemas de Organização
        6. Dificuldade com Prazos
        """)
        
        st.header("Desenvolvedores")
        st.markdown("""
        - Victor Gabriel Cunha Rodrigues
        - Lorrana Evelyn de Araújo Pereira
        """)
        
        st.header("Universidade Federal do Piauí")
        st.markdown("Curso de Tecnologia de Gestão de Dados")
    
    # Área de entrada de texto
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            texto_input = st.text_area(
                "Digite o relato do estudante:",
                height=200,
                placeholder="Ex: Estou tendo muita dificuldade para entender os textos longos das aulas de história..."
            )
            
        with col2:
            st.markdown("**Exemplos para testar:**")
            st.markdown("- 'Não consigo resolver equações de segundo grau'")
            st.markdown("- 'Fico muito nervoso nas provas e esqueço tudo'")
            st.markdown("- 'Sempre deixo tudo para a última hora'")
    
    # Botão de classificação
    if st.button("Classificar Dificuldade", use_container_width=True):
        if not texto_input.strip():
            st.warning("Por favor, insira um texto para classificação.")
        elif modelo and vectorizer and categorias:
            with st.spinner("Analisando o texto..."):
                # Pré-processamento
                texto_processado = preprocessar_texto(texto_input)
                
                # Vetorização
                texto_vetorizado = vectorizer.transform([texto_processado])
                
                # Predição
                probabilidades = modelo.predict_proba(texto_vetorizado)[0]
                predicao = np.argmax(probabilidades)
                
                # Resultados
                st.subheader("Resultado da Classificação")
                
                # Gráfico de probabilidades
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    x=probabilidades, 
                    y=list(categorias.values()),
                    palette="viridis",
                    ax=ax
                )
                ax.set_title("Probabilidade por Categoria")
                ax.set_xlabel("Probabilidade")
                ax.set_ylabel("Categoria")
                st.pyplot(fig)
                
                # Detalhes da predição
                st.success(f"**Categoria mais provável:** {categorias[predicao]}")
                
                # Tabela de probabilidades
                prob_df = pd.DataFrame({
                    "Categoria": categorias.values(),
                    "Probabilidade": probabilidades
                }).sort_values("Probabilidade", ascending=False)
                
                st.dataframe(
                    prob_df.style.format({"Probabilidade": "{:.2%}"}),
                    use_container_width=True
                )
                
                # Sugestões baseadas na categoria
                st.subheader("Sugestões de Apoio")
                if predicao == 0:  # Leitura
                    st.markdown("""
                    - Praticar leitura diária de textos curtos
                    - Usar marcadores de texto para identificar ideias principais
                    - Participar de grupos de estudo de interpretação
                    """)
                elif predicao == 1:  # Matemática
                    st.markdown("""
                    - Revisar conceitos básicos que são pré-requisitos
                    - Fazer exercícios gradativamente mais complexos
                    - Usar aplicativos de aprendizagem matemática
                    """)
                elif predicao == 2:  # Ansiedade
                    st.markdown("""
                    - Praticar técnicas de respiração antes de provas
                    - Dividir tarefas grandes em partes menores
                    - Buscar apoio psicológico se necessário
                    """)
                elif predicao == 3:  # Concentração
                    st.markdown("""
                    - Estudar em ambientes silenciosos e organizados
                    - Usar técnicas Pomodoro (25min estudo + 5min descanso)
                    - Evitar multitarefas durante o estudo
                    """)
                elif predicao == 4:  # Organização
                    st.markdown("""
                    - Usar agendas ou aplicativos de planejamento
                    - Criar listas de tarefas diárias
                    - Estabelecer rotinas de estudo fixas
                    """)
                elif predicao == 5:  # Prazos
                    st.markdown("""
                    - Dividir trabalhos grandes em etapas com prazos intermediários
                    - Começar tarefas imediatamente após serem passadas
                    - Usar lembretes para entregas importantes
                    """)
        else:
            st.error("Modelo não carregado corretamente. Por favor, verifique os arquivos de modelo.")

    # Seção para análise de arquivos CSV (para uso dos professores)
    st.divider()
    st.subheader("Análise em Lote (para Professores)")
    
    arquivo = st.file_uploader(
        "Carregue um arquivo CSV com múltiplos relatos para análise em lote",
        type=["csv"],
        accept_multiple_files=False
    )
    
    if arquivo is not None:
        try:
            df = pd.read_csv(arquivo)
            
            if 'texto' not in df.columns:
                st.warning("O arquivo CSV deve conter uma coluna chamada 'texto' com os relatos.")
            else:
                with st.spinner("Processando múltiplos relatos..."):
                    # Pré-processamento
                    df['texto_processado'] = df['texto'].apply(preprocessar_texto)
                    
                    # Vetorização
                    X = vectorizer.transform(df['texto_processado'])
                    
                    # Predição
                    probabilidades = modelo.predict_proba(X)
                    predicoes = np.argmax(probabilidades, axis=1)
                    
                    # Adicionar resultados ao DataFrame
                    df['categoria'] = [categorias[p] for p in predicoes]
                    df['categoria_num'] = predicoes
                    
                    # Mostrar resultados
                    st.success(f"Processamento concluído para {len(df)} relatos!")
                    
                    # Gráfico de distribuição
                    st.subheader("Distribuição das Categorias")
                    contagem = df['categoria'].value_counts()
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    sns.barplot(
                        x=contagem.values,
                        y=contagem.index,
                        palette="rocket",
                        ax=ax2
                    )
                    ax2.set_title("Quantidade de Relatos por Categoria")
                    ax2.set_xlabel("Número de Relatos")
                    ax2.set_ylabel("Categoria")
                    st.pyplot(fig2)
                    
                    # Tabela com resultados
                    st.dataframe(
                        df[['texto', 'categoria']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Botão para download
                    st.download_button(
                        label="Baixar Resultados",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name="relatos_classificados.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

if __name__ == "__main__":
    main()