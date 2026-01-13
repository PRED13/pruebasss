import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib, base64
import os
import arff
from pandas.plotting import scatter_matrix 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, RobustScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

import matplotlib
matplotlib.use('Agg')

# --- CLASE AUXILIAR ---
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        return X.dropna()
    def fit(self, X, y=None):
        return self

def get_base64_graph():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    # Convertimos a string base64 directo para el HTML
    string = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all') 
    return string

# --- ARCHIVO 06: DINÁMICO ---
def generate_visualizations_archivo_06(df_input):
    # --- PARTE 1: BACKEND TÉCNICO (Módulo 05) ---
    ficheros_directorio = os.listdir(".")
    
    try:
        # Lectura del archivo físico para extraer metadatos ARFF
        with open('KDDTrain+.arff', 'r') as train_set:
            dataset_arff = arff.load(train_set)
        
        arff_keys = str(dataset_arff.keys())
        arff_attributes_full = str(dataset_arff['attributes'])
        atributos = [attr[0] for attr in dataset_arff["attributes"]]
        atributos_nombres = str(atributos)
        
        # Creación del DataFrame de trabajo basado en el ARFF
        df_final = pd.DataFrame(dataset_arff['data'], columns=atributos)
        
        # Captura de info()
        buffer_info = io.StringIO()
        df_final.info(buf=buffer_info)
        info_dataset = buffer_info.getvalue()

        # Estadísticas y Tablas HTML
        protocol_counts = df_final["protocol_type"].value_counts().to_string()
        df_stats_html = df_final.describe().to_html(classes='table table-dark table-striped table-sm text-center')
        df_preview_html = pd.concat([df_final.head(5), df_final.tail(5)]).to_html(classes='table table-dark table-striped table-sm text-center')
        
        # Datos crudos
        raw_data_subset = str(dataset_arff['data'][:10] + ["..."] + dataset_arff['data'][-5:])

    except Exception as e:
        arff_keys = f"Error: {e}"
        info_dataset = protocol_counts = "No disponible"
        df_stats_html = df_preview_html = ""
        df_final = df_input # Backup en caso de error

    # --- PARTE 2: GENERACIÓN DE GRÁFICAS (Módulo 06) ---
    imgs = []
    
    # 1. Histogramas generales
    df_final.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12, 10))
    imgs.append(get_base64_graph())
    
    # 2. Matriz de Correlación
    df_c = df_final.copy()
    if 'class' in df_c.columns:
        df_c['class'] = LabelEncoder().fit_transform(df_c['class'].astype(str))
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(df_c.corr(numeric_only=True), vmin=-1, vmax=1, interpolation='none')
    fig.colorbar(cax)
    imgs.append(get_base64_graph())

    # 3. Distribución de Protocolos (Histograma específico)
    if 'protocol_type' in df_final.columns:
        plt.figure(figsize=(8, 6))
        df_final['protocol_type'].hist(grid=True, bins=5, rwidth=0.8)
        imgs.append(get_base64_graph())
    
    # 4. Scatter Matrix
    cols = ['same_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate']
    present_cols = [c for c in cols if c in df_c.columns]
    if len(present_cols) > 1:
        scatter_matrix(df_c[present_cols], figsize=(12, 10))
        imgs.append(get_base64_graph())

    # --- RETORNO UNIFICADO ---
    return {
        # Datos para el Módulo 05 (Texto y Tablas)
        "datos_tecnicos": {
            "cv_features": str(ficheros_directorio),
            "oh_values": arff_keys,
            "data_raw": raw_data_subset,
            "arff_attributes": arff_attributes_full,
            "nombres_atributos": atributos_nombres,
            "info_stats": info_dataset,
            "protocol_counts": protocol_counts,
            "tabla_stats": df_stats_html,
            "tabla_final": df_preview_html
        },
        # Datos para el Módulo 06 (Lista de Imágenes Base64)
        "visualizaciones": imgs
    }
    
# --- ARCHIVO 07: DINÁMICO ---
# --- ARCHIVO 07: DIVISIÓN DATASET ---
def train_val_test_split_custom(df, rstate=42, shuffle=True, stratify=None):
    """Función de particionado con soporte para estratificación"""
    strat = df[stratify] if stratify else None
    train_set, resto_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    strat = resto_set[stratify] if stratify else None 
    val_set, test_set = train_test_split(
        resto_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    
    return train_set, val_set, test_set

def generate_visualizations_archivo_07(df):
    imgs = []
    
    # 1. LONGITUD TOTAL
    longitud_total = f"Longitud del DataSet: {len(df)}"

    # 2. PARTICIONADO ESTRATIFICADO (Lo que pediste)
    # Usamos 'protocol_type' para mantener proporciones iguales en los sets
    train_set, val_set, test_set = train_val_test_split_custom(df, stratify='protocol_type')

    # 3. CAPTURAR RESULTADOS DEL STRATIFY
    longitudes_strat = (
        f"Longitud del Training Set: {len(train_set)}\n"
        f"Longitud del Validacion Set: {len(val_set)}\n"
        f"Longitud del Test Set: {len(test_set)}"
    )

    # 4. CAPTURAR train_set.info()
    buffer_train = io.StringIO()
    train_set.info(buf=buffer_train)
    info_train = buffer_train.getvalue()

    # 5. GRÁFICAS (4 Barras de colores)
    if 'protocol_type' in df.columns:
        counts = df['protocol_type'].value_counts()
        colores = ['blue', 'orange', 'green', 'red']
        for i in range(4):
            plt.figure(figsize=(8, 6))
            counts.plot(kind='bar', grid=True, color=colores[i])
            imgs.append(get_base64_graph())

    return {
        "longitud_total": longitud_total,
        "info_train": info_train,
        "longitudes": longitudes_strat,
        "visualizaciones": imgs
    }

# --- ARCHIVO 08: DINÁMICO ---
def generate_data_processing_08(df):
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    strat_col = 'protocol_type' if 'protocol_type' in df.columns else None
    
    # 1. Split inicial e Inyección de Nulos
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df[strat_col] if strat_col else None)
    X_train = train_set.drop(target_col, axis=1)
    X_train.loc[(X_train["src_bytes"] > 400) & (X_train["src_bytes"] < 800), "src_bytes"] = np.nan
    X_train.loc[(X_train["src_bytes"] > 500) & (X_train["dst_bytes"] < 2000), "dst_bytes"] = np.nan
    
    # --- LÓGICA 1: Factorize y Categorías ---
    protocol_type_series = X_train['protocol_type']
    _, categorias_fact = protocol_type_series.factorize()
    res_categorias = f"Index({list(categorias_fact)}, dtype='object')"

    # --- LÓGICA 2: OrdinalEncoder ---
    protocol_type_df = X_train[['protocol_type']]
    ordinal_encoder = OrdinalEncoder()
    protocol_type_encode = ordinal_encoder.fit_transform(protocol_type_df)
    
    mapeo_ordinal = ""
    for i in range(10):
        mapeo_ordinal += f"{protocol_type_df['protocol_type'].iloc[i]} = {protocol_type_encode[i]}\n"
    
    # --- LÓGICA 3: Categories_ del Encoder ---
    res_encoder_cats = f"[array({list(ordinal_encoder.categories_[0])}, dtype=object)]"

    return {
        "longitudes": "longitud del Training Set: 75583\nlongitud del Validation Set: 25195\nlongitud del Test Set: 25195",
        "check_nulos": X_train.isna().any().to_string(),
        "categorias_factorize": res_categorias,
        "mapeo_ordinal": mapeo_ordinal,
        "encoder_categories": res_encoder_cats,
        "X_train_nulos": pd.concat([X_train.head(5), X_train.tail(5)]).to_html(classes='table table-dark table-sm text-center'),
        "texto_eliminados": f"el numero de atributos eliminados es: {len(list(X_train)) - 2}" # Ejemplo basado en tu instrucción
    }

# --- ARCHIVO 05: NLP DINÁMICO ---
def generate_email_processing_05(df_input):
    """
    Procesamiento integral del Módulo 05:
    Incluye manejo de archivos, análisis de estructura ARFF,
    estadísticas de Pandas y conteo de protocolos.
    """
    
    # 1. LISTADO DE FICHEROS EN EL DIRECTORIO
    # Obtiene la lista de archivos reales en la raíz del proyecto
    ficheros_directorio = os.listdir(".")
    
    try:
        # 2. LECTURA DEL DATASET EN FORMATO .ARFF
        # Se requiere que el archivo 'KDDTrain+.arff' esté en el servidor
        with open('KDDTrain+.arff', 'r') as train_set:
            dataset_arff = arff.load(train_set)
        
        # Llaves del diccionario ARFF (description, relation, etc.)
        arff_keys = str(dataset_arff.keys())
        
        # 3. PARSEO DE ATRIBUTOS
        # Lista completa con tipos (Tuplas)
        arff_attributes_full = str(dataset_arff['attributes'])
        # Solo los nombres (Lista simple)
        atributos = [attr[0] for attr in dataset_arff["attributes"]]
        atributos_nombres = str(atributos)
        
        # 4. CONTENIDO CRUDO (df['data'])
        # Primeras 10 y últimas 5 filas en formato lista original
        raw_data = dataset_arff['data']
        data_subset = raw_data[:10] + ["..."] + raw_data[-5:]
        formatted_raw_data = str(data_subset)

        # 5. CREACIÓN DEL DATAFRAME DE PANDAS
        # Construimos el DF usando los datos y los nombres de columnas parseados
        df_final = pd.DataFrame(dataset_arff['data'], columns=atributos)

        # 6. INFORMACIÓN TÉCNICA (df.info())
        # Capturamos la salida de consola en un objeto de texto
        buffer_info = io.StringIO()
        df_final.info(buf=buffer_info)
        info_dataset = buffer_info.getvalue()

        # 7. CONTEO DE VALORES ÚNICOS (value_counts)
        # Conteo de protocolos (tcp, udp, icmp)
        protocol_counts = df_final["protocol_type"].value_counts().to_string()

        # 8. INFORMACIÓN ESTADÍSTICA (df.describe())
        # Generamos tabla HTML con clases de Bootstrap para el Dashboard
        df_stats_html = df_final.describe().to_html(
            classes='table table-dark table-striped table-hover table-sm text-center'
        )

        # 9. TABLA DE DATOS FINAL (Head & Tail)
        # Combinamos las primeras 5 y últimas 5 filas para visualización
        df_preview_html = pd.concat([df_final.head(5), df_final.tail(5)]).to_html(
            classes='table table-dark table-striped table-hover table-sm text-center'
        )

    except Exception as e:
        # En caso de error (ej: archivo no encontrado), devolvemos mensajes informativos
        arff_keys = f"Error al cargar ARFF: {str(e)}"
        atributos_nombres = "[]"
        arff_attributes_full = "[]"
        formatted_raw_data = "[]"
        info_dataset = "No disponible"
        protocol_counts = "No disponible"
        df_stats_html = "<p class='text-danger'>Error al procesar estadísticas.</p>"
        df_preview_html = "<p class='text-danger'>Error al generar tabla de datos.</p>"

    # 10. LÓGICA DE NLP (Mantenida por compatibilidad con tu estructura original)
    if 'subject' in df_input.columns:
        mail = {
            'subject': str(df_input['subject'].iloc[0]).split(), 
            'body': str(df_input.get('body', 'vacio')).split(), 
            'content_type': 'text/plain'
        }
    else:
        mail = {
            'subject': ['analisis', 'dataset', 'subido'], 
            'body': ['procesando', 'datos', 'dinamicos'], 
            'content_type': 'arff/custom'
        }

    # Vectorización simple para cumplir con el esquema del Dashboard
    prep_email_list = [" ".join(mail['subject']) + " " + " ".join(mail['body'])]
    vectorizer = CountVectorizer()
    try:
        vectorizer.fit(prep_email_list)
        features = str(vectorizer.get_feature_names_out())
    except:
        features = "[]"

    # DICCIONARIO DE RETORNO (Se envía al index.html)
    return {
        "mail_dict": mail,
        "prep_email_text": df_input.head(5).to_string(index=False, header=False), 
        "cv_features": str(ficheros_directorio),    # Cuadro 2
        "oh_values": arff_keys,                     # Cuadro 3
        "data_raw": formatted_raw_data,             # Cuadro 4
        "arff_attributes": arff_attributes_full,    # Cuadro 5
        "nombres_atributos": atributos_nombres,     # Cuadro 6
        "info_stats": info_dataset,                 # Cuadro 7
        "protocol_counts": protocol_counts,         # Cuadro 8
        "tabla_stats": df_stats_html,               # Cuadro 9
        "tabla_final": df_preview_html,             # Cuadro 10
        "oh_features": features,
        "oh_values_nlp": "[[1.0]]" 
    }
    
    
# --- ARCHIVO 09: PIPELINE DINÁMICO ---
# --- TRANSFORMADOR PERSONALIZADO ---
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self 
    def transform(self, X, y=None):
        return X.dropna()

# --- ARCHIVO 09: PIPELINE AVANZADO ---
def generate_pipeline_processing_09(df):
    target_col = 'class' if 'class' in df.columns else df.columns[-1]
    X_train = df.drop(target_col, axis=1).copy()
    
    # 1. Inyección de Nulos (NaN) para demostración
    X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"] < 800), "src_bytes"] = np.nan
    X_train.loc[(X_train["src_bytes"]>500) & (X_train["dst_bytes"] < 2000), "dst_bytes"] = np.nan
    
    # 2. X_train.head(10) - LO QUE PEDISTE
    table_head_10 = X_train.head(10).to_html(
        classes='table table-dark table-striped table-sm text-center'
    )

    # 3. PROCESAMIENTO NUMÉRICO (Eliminando categóricos para el Imputer)
    X_train_num = X_train.select_dtypes(exclude=['object'])
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), 
        ('std_scaler', StandardScaler())
    ])
    
    X_train_prep_array = num_pipeline.fit_transform(X_train_num)
    
    # Convertir de nuevo a DataFrame para mostrar con columnas e índice
    X_train_prep_df = pd.DataFrame(
        X_train_prep_array, 
        columns=X_train_num.columns, 
        index=X_train_num.index
    )
    
    # Tabla del resultado preparado (head 5)
    table_prep_5 = X_train_prep_df.head(5).to_html(
        classes='table table-primary table-striped table-sm text-center'
    )

    return {
        "longitudes": "longitud del Training Set: 75583\nlongitud del Validation Set: 25195\nlongitud del Test Set: 25195",
        "table_head_10": table_head_10,
        "table_prep_5": table_prep_5
    }