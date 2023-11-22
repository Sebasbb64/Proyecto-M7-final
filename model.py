import joblib
import nltk
import re
import unidecode
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

STOP_WORDS_EN = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()

model = joblib.load('Proyecto_modulo_7_final.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def get_prediction(text):
    text = clean_text(text)
    vectorized_text = vectorizer.transform([text])
    sentimiento = model.predict(vectorized_text)[0]
    sentimiento_proba = model.predict_proba(vectorized_text)[0]   

    if sentimiento == 1:
        sentimiento = 'La opinión tiene un sentimiento: Positivo'
    elif sentimiento == -1:
        sentimiento = 'La opinión tiene un sentimiento: Negativo'
    elif sentimiento == 0:
        sentimiento = 'La opinión tiene un sentimiento: Neutral'
    else:
        sentimiento = 'La opinión tiene un sentimiento no identificado'

    return sentimiento, sentimiento_proba

def clean_text(text):

    #Convetir a minúscula
    text2 = text.lower()

    #Quitar acentos
    text2 = unidecode.unidecode(text2)


    text2 = re.sub('[^a-zA-Z]', ' ', text2)

    #Remover puntuación
    text2 =re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text2)

    #Remover dígitos y carácteres especiales
    text2 = re.sub("(\\d|\\W)+"," ",text2)

    # Removemos tags HTML
    text2 = re.sub(re.compile('((http|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?)'), '', text2)

    # Tomamos solo las palabras
    text2 = re.sub('[^A-Za-z0-9]+', ' ', text2)

    # Tokenización
    text2 = nltk.word_tokenize(text2)

    # Removemos las palabras de parada
    text2 = [word for word in text2 if word not in STOP_WORDS_EN]

    # Lematizamos
    text2 = [lemmatizer.lemmatize(token, pos="v") for token in text2]
    text2 = [lemmatizer.lemmatize(token) for token in text2]

    # Unimos las palabras
    text2 = ' '.join(text2)

    return text2
