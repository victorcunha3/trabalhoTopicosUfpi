import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
from processamento import preprocessar_texto

def criar_dataset_simulado():
    """
    Cria um dataset simulado com relatos de estudantes e suas categorias
    """
    dados = {
        'texto': [
            'não consigo entender os textos longos das aulas',
            'sempre erro as contas de multiplicação e divisão',
            'fico muito nervoso quando tenho que apresentar trabalhos',
            'me distraio facilmente com qualquer barulho na sala',
            'não sei como organizar meu tempo para estudar todas as matérias',
            'sempre deixo os trabalhos para a última hora',
            'ler livros inteiros é muito difícil para mim',
            'equações de segundo grau são um mistério',
            'minha mente fica em branco nas provas mesmo sabendo a matéria',
            'perco o foco quando estudo em casa',
            'minha mochila e cadernos são uma bagunça',
            'nunca lembro dos prazos de entrega',
            'interpretação de texto é meu ponto fraco',
            'frações são complicadas demais',
            'tenho medo de tirar notas baixas',
            'não consigo prestar atenção em aulas longas',
            'meus materiais estão sempre desorganizados',
            'sempre preciso de lembretes para entregar trabalhos'
        ],
        'categoria': [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    }
    
    return pd.DataFrame(dados)

def treinar_modelo():
    # Criar dataset simulado
    df = criar_dataset_simulado()
    
    # Pré-processamento
    df['texto_processado'] = df['texto'].apply(preprocessar_texto)
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        df['texto_processado'], 
        df['categoria'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Vetorização (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Treinar modelo (Naive Bayes)
    modelo = MultinomialNB()
    modelo.fit(X_train_vec, y_train)
    
    # Avaliação
    y_pred = modelo.predict(X_test_vec)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Salvar modelo e vectorizer
    joblib.dump(modelo, 'main/models/modelo_naive_bayes.pkl')
    joblib.dump(vectorizer, 'main/models/vectorizer.pkl')
    
    print("Modelo e vetorizador salvos com sucesso!")

if __name__ == "__main__":
    treinar_modelo()