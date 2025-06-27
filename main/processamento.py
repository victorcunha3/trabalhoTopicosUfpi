import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# Baixar recursos do NLTK (executar apenas na primeira vez)
nltk.download('stopwords')
nltk.download('rslp')

# Carregar stopwords em português
stop_words = set(stopwords.words('portuguese'))

# Inicializar stemmer
stemmer = RSLPStemmer()

def preprocessar_texto(texto):
    """
    Realiza o pré-processamento de um texto em português:
    1. Conversão para minúsculas
    2. Remoção de pontuação
    3. Remoção de números
    4. Remoção de stopwords
    5. Stemming
    6. Remoção de espaços extras
    """
    if not isinstance(texto, str):
        return ""
    
    # Converter para minúsculas
    texto = texto.lower()
    
    # Remover pontuação
    texto = re.sub(f'[{string.punctuation}]', ' ', texto)
    
    # Remover números
    texto = re.sub(r'\d+', '', texto)
    
    # Remover stopwords
    palavras = [palavra for palavra in texto.split() if palavra not in stop_words]
    
    # Aplicar stemming
    palavras = [stemmer.stem(palavra) for palavra in palavras]
    
    # Juntar palavras novamente
    texto_processado = ' '.join(palavras)
    
    # Remover espaços extras
    texto_processado = re.sub(r'\s+', ' ', texto_processado).strip()
    
    return texto_processado