import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

nltk.download('stopwords')
nltk.download('punkt')

nltk.download('stopwords')

# Cargar datos
df = pd.read_csv("opiniones_clientes.csv")
opiniones = df['Opinion'].astype(str).tolist()

# Procesamiento
texto = " ".join(opiniones).lower()
texto = re.sub(r'[^\w\s]', '', texto)
stopwords_es = set(stopwords.words('spanish'))
palabras = [w for w in texto.split() if w not in stopwords_es and len(w) > 2]

# Mostrar título
st.title("📝 Análisis de Opiniones de Clientes")

# Nube de palabras
if st.button("Mostrar Nube de Palabras"):
    nube = WordCloud(width=800, height=400, background_color='white').generate(" ".join(palabras))
    st.image(nube.to_array(), use_column_width=True)

# Top palabras
if st.button("Mostrar Top 10 Palabras"):
    conteo = Counter(palabras)
    comunes = conteo.most_common(10)
    etiquetas, valores = zip(*comunes)
    fig, ax = plt.subplots()
    ax.bar(etiquetas, valores, color='pink')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Análisis de sentimientos
modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(modelo)
model = AutoModelForSequenceClassification.from_pretrained(modelo)
clasificador = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def interpretar(label):
    estrellas = int(label[0])
    return "Positivo" if estrellas >= 4 else "Neutro" if estrellas == 3 else "Negativo"

if st.button("Mostrar Sentimientos de Comentarios"):
    resultados = clasificador(opiniones)
    df['Sentimiento'] = [interpretar(r['label']) for r in resultados]
    conteo = df['Sentimiento'].value_counts()
    st.dataframe(df[['Opinion', 'Sentimiento']])
    fig2, ax2 = plt.subplots()
    ax2.pie(conteo, labels=conteo.index, autopct='%1.1f%%', colors=['pink', 'violet', 'skyblue'])
    st.pyplot(fig2)

# Comentario nuevo
st.subheader("🗣 Escribe un comentario nuevo")
comentario = st.text_input("Tu comentario:")

if st.button("Analizar Comentario"):
    if comentario:
        resultado = clasificador(comentario)[0]
        st.write("✅ Sentimiento:", interpretar(resultado['label']))
        palabras_clave = [w for w in re.findall(r'\b\w+\b', comentario.lower()) if w not in stopwords_es]
        st.write("🔍 Palabras clave:", ", ".join(palabras_clave[:5]) if palabras_clave else "No se encontraron palabras clave")
    else:
        st.warning("Por favor escribe un comentario.")

