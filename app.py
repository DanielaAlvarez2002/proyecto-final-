import base64
import io
import pandas as pd
import dash
from dash import html, dcc, dash_table
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
import spacy
from collections import Counter
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("es_core_news_sm")

ruta = "Base de datos de reseñas.csv"
df = pd.read_csv(ruta, encoding='utf-8', sep=';')
df = df.iloc[1:26].copy()

stop_words_es = set(stopwords.words('spanish'))
stop_words_en = set(stopwords.words('english'))
conectores = {
    'on', 'with', 'in', 'at', 'of', 'to', 'from', 'up', 'down', 'out', 'off',
    'this', 'that', 'these', 'those', 'it', 'its', 'and', 'but', 'or', 'so', 'very'
}
stop_words_total = stop_words_es.union(stop_words_en).union(conectores)

def limpiar_y_lemmatizar(texto):
    if pd.isna(texto):
        return ""
    texto = texto.lower()
    texto = ''.join([c for c in texto if c not in string.punctuation])
    doc = nlp(texto)
    palabras = [
        token.lemma_ for token in doc
        if token.pos_ == "ADJ"
        and token.lemma_ not in stop_words_total
        and len(token.lemma_) > 2
    ]
    return ' '.join(palabras)

df['texto_limpio'] = df['Review.Text'].astype(str).apply(limpiar_y_lemmatizar)


def obtener_sentimiento(texto):
    blob = TextBlob(texto)
    pol = blob.sentiment.polarity
    if pol > 0.3:
        return "Positivo"
    elif pol < -0.3:
        return "Negativo"
    else:
        return "Neutro"

df['Sentimiento'] = df['Review.Text'].astype(str).apply(obtener_sentimiento)
sent_counts = df['Sentimiento'].value_counts().reset_index()
sent_counts.columns = ['Sentimiento', 'Cantidad']

adjetivos = []
for texto in df['Review.Text'].astype(str):
    doc = nlp(texto.lower())
    adjetivos.extend([
        token.lemma_ for token in doc
        if token.pos_ == "ADJ"
        and token.lemma_ not in stop_words_total
        and len(token.lemma_) > 2
    ])

texto_adjetivos = ' '.join(adjetivos)

wc = WordCloud(
    width=800, height=400,
    background_color='white',
    colormap='winter'
).generate(texto_adjetivos)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
wc_buffer = io.BytesIO()
plt.savefig(wc_buffer, format='png')
plt.close()
wc_buffer.seek(0)
encoded_wc = base64.b64encode(wc_buffer.read()).decode('utf-8')

palabras = texto_adjetivos.split()
frecuencias = Counter(palabras).most_common(10)
df_frec = pd.DataFrame(frecuencias, columns=['Palabra', 'Frecuencia'])

fig_frec = px.bar(
    df_frec,
    x='Palabra',
    y='Frecuencia',
    text='Frecuencia',
    title='Top 10 adjetivos más frecuentes',
    color='Frecuencia',
    color_continuous_scale=['#1f77b4', '#2ca02c']  # azul a verde
)
fig_frec.update_traces(textposition='outside')
fig_frec.update_layout(plot_bgcolor='white', title_font_color='#1f77b4')

fig_sent = px.pie(
    sent_counts,
    values='Cantidad',
    names='Sentimiento',
    color='Sentimiento',
    color_discrete_map={'Positivo': 'green', 'Negativo': 'red', 'Neutro': 'gray'},
    title='Distribución de sentimientos (filas 2–25)'
)
fig_sent.update_traces(textinfo='percent+label', pull=[0.05, 0.05, 0])

if 'Recommended.IND' in df.columns:
    df['Recommended.IND'] = df['Recommended.IND'].map({1: 'Recomienda', 0: 'No Recomienda'})
    df_rec = df.groupby(['Recommended.IND', 'Sentimiento']).size().reset_index(name='Cantidad')
    fig_rec = px.bar(
        df_rec,
        x='Recommended.IND',
        y='Cantidad',
        color='Sentimiento',
        barmode='group',
        title='Relación entre Sentimiento y Recomendación',
        color_discrete_map={'Positivo': '#2ca02c', 'Negativo': 'red', 'Neutro': 'gray'}
    )
    fig_rec.update_layout(plot_bgcolor='white', title_font_color='#2ca02c')
else:
    fig_rec = px.bar(title='Columna "Recommended.IND" no encontrada')

tabla_sent = dash_table.DataTable(
    columns=[{"name": i, "id": i} for i in ['Review.Text', 'Sentimiento']],
    data=df[['Review.Text', 'Sentimiento']].to_dict('records'),
    page_size=5,
    style_table={'overflowX': 'auto', 'maxHeight': '400px'},
    style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
    style_header={'backgroundColor': '#1f77b4', 'color': 'white', 'fontWeight': 'bold'},
    style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#e6f2ec'}]
)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Análisis de Opiniones de Clientes", style={'textAlign': 'center', 'color': '#1f77b4'}),

    html.Div([
        html.Div([
            html.H4("Nube de Palabras (solo adjetivos)", style={'textAlign': 'center', 'color': '#2ca02c'}),
            html.Img(src='data:image/png;base64,{}'.format(encoded_wc),
                     style={'width': '100%', 'display': 'block', 'margin': 'auto'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H4("Frecuencia de Adjetivos", style={'textAlign': 'center', 'color': '#2ca02c'}),
            dcc.Graph(figure=fig_frec)
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    ], style={'padding': '20px'}),


    html.Div([
        html.Div([
            html.H4("Análisis de Sentimientos", style={'textAlign': 'center', 'color': '#1f77b4'}),
            dcc.Graph(figure=fig_sent)
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H4("Tabla de Reseñas y Sentimiento", style={'textAlign': 'center', 'color': '#1f77b4'}),
            tabla_sent
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    ], style={'padding': '20px'}),
    
    html.Div([
        html.Div([
            html.H4("Relación entre Sentimiento y Recomendación", style={'textAlign': 'center', 'color': '#2ca02c'}),
            dcc.Graph(figure=fig_rec)
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H3("Analiza tu propio comentario", style={'textAlign': 'center', 'color': '#2ca02c'}),
            dcc.Textarea(
                id='input-comentario',
                placeholder='Escribe aquí tu opinión...',
                style={'width': '90%', 'height': 100, 'margin': 'auto', 'display': 'block',
                       'borderRadius': '10px', 'border': '2px solid #1f77b4'}
            ),
            html.Br(),
            html.Button(
                'Analizar sentimiento', id='boton-analizar', n_clicks=0,
                style={'display': 'block', 'margin': 'auto',
                       'backgroundColor': '#2ca02c', 'color': 'white',
                       'border': 'none', 'padding': '10px 20px', 'borderRadius': '8px',
                       'fontWeight': 'bold'}
            ),
            html.Br(),
            html.Div(id='resultado-sentimiento',
                     style={'textAlign': 'center', 'fontSize': '18px',
                            'fontWeight': 'bold', 'color': '#1f77b4'})
        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    ], style={'padding': '20px'})
])

@app.callback(
    dash.Output('resultado-sentimiento', 'children'),
    dash.Input('boton-analizar', 'n_clicks'),
    dash.State('input-comentario', 'value')
)
def analizar_sentimiento_usuario(n_clicks, comentario):
    if n_clicks == 0 or not comentario:
        return ""
    blob = TextBlob(comentario)
    pol = blob.sentiment.polarity
    if pol > 0.3:
        return "Reseña positiva"
    elif pol < -0.3:
        return "Reseña negativa"
    else:
        return "Reseña neutra"

if __name__ == '__main__':
    app.run(debug=True)

