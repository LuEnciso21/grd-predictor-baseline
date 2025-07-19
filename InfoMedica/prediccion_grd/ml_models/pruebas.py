import pandas as pd
import numpy as np
import joblib
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Definir paths para cargar modelo, scaler y columnas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, 'modelo_grd.pkl')
COLUMNAS_PATH = os.path.join(BASE_DIR, 'columnas_modelo.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_modelo.pkl')

# Cargar modelo, scaler y columnas esperadas (solo variables input)
modelo = joblib.load(MODELO_PATH)
scaler = joblib.load(SCALER_PATH)

with open(COLUMNAS_PATH, 'rb') as f:
    columnas = pickle.load(f)  # Lista ordenada de columnas que espera el modelo (sin etiquetas)

def codificar_variables(df):
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns
    for col in columnas_categoricas:
        valores_unicos = df[col].dropna().unique()
        if len(valores_unicos) == 2:
            # Mapear binario a 0 y 1
            df[col] = df[col].map({valores_unicos[0]: 0, valores_unicos[1]: 1})
        else:
            # Mapear frecuencia para múltiples categorías
            frecuencia = df[col].value_counts()
            df[col] = df[col].map(frecuencia)
    return df

def alinear_columnas(df, columnas_esperadas):
    for col in columnas_esperadas:
        if col not in df.columns:
            print(f"[WARN] Columna faltante: {col}. Se añade con ceros.")
            df[col] = 0
    # Ordenar columnas exactamente como el modelo espera
    return df[columnas_esperadas]

def preparar_datos_para_modelo(df_input):
    df = df_input.copy()
    
    # Limpiar espacios
    df.columns = df.columns.str.strip()
    
    # Codificar categóricas
    df = codificar_variables(df)
    
    # Rellenar NaN
    df.fillna(0, inplace=True)
    
    # Alinear columnas con las que espera el modelo (agrega columnas faltantes con 0)
    df = alinear_columnas(df, columnas)
    
    # Filtrar columnas numéricas entre las columnas completas del modelo (orden fijo)
    columnas_numericas_modelo = [col for col in columnas if pd.api.types.is_numeric_dtype(df[col])]
    
    # Añadir epsilon SOLO en esas columnas numéricas (orden correcto)
    epsilon = 1e-3
    df.loc[:, columnas_numericas_modelo] += epsilon
    
    # Aplicar scaler en esas columnas EXACTAS, mismo orden
    df.loc[:, columnas_numericas_modelo] = scaler.transform(df.loc[:, columnas_numericas_modelo])
    
    return df

# Ejemplo de datos de entrada (puedes usar tus propios datos)
data_cruda = {
    'NUM CASO': [413686],
    'ASEGURADORA -CODIGO-': ['227'],
    'ASEGURADORA -DESCRIPCION-': ['Aseguradora_46'],
    'EDAD': [100],
    'GRUPO EDAD': ['> 90'],
    'SEXO': ['M'],
    'CODIGO DE CIUDAD': ['5001'],
    'FECHA DE INGRESO': ['25052005'],
    'TIPO DE INGRESO': ['URGENCIA'],
    'FECHA DE EGRESO': ['29052005'],
    'DIAS ESTANCIA': [4],
    'SERVICIOALTA': ['54'],
    'CUIDADOS INTENSIVOS': ['NO'],
    'DIAS DE UNIDAD CUIDADO INTENSIVO': [0],
    'DX DE INGRESO': ['2'],
    'DX PRINCIPAL DE EGRESO': ['N178'],
    'DX PRINCIPAL DE EGRESO .1': ['N17'],
    'DX PPAL 3 CARACTERES': ['INSUFICIENCIA RENAL AGUDA'],
    'DXR 1': ['G309'],
    'DXR 2': ['F009'],
    'DXR 3': ['F051'],
    'DXR 4': ['E46'],
    'DXR 5': ['E038'],
    'DXR-6': ['D648'],
    'SITUACION AL ALTA': ['ALTA MÉDICA'],
    'PROC1': ['45,16'],
    'PROC2': ['87,44'],
    'PROC3': ['87,03'],
    'TIPO SERVICIO': ['URGENCIA ADULTOS'],
    'INFECCIONES': ['NO'],
    'INFECCION QUIRURGICA': ['NO'],
    'TIPO GRD':['M']
}

df_input = pd.DataFrame(data_cruda)

# Convertir comas a puntos y pasar a float en columnas PROC si existen
for col in df_input.columns:
    if 'PROC' in col:
        df_input[col] = df_input[col].astype(str).str.replace(',', '.').astype(float)

# Preparar datos para el modelo
df_preparado = preparar_datos_para_modelo(df_input)

# Predicción
prediccion = modelo.predict(df_preparado)[0]
probabilidades = modelo.predict_proba(df_preparado)[0]
grds = modelo.classes_
prob_grds_sorted = sorted(zip(grds, probabilidades), key=lambda x: x[1], reverse=True)

print(f'GRD Predicho: {prediccion}')
print('Top 3 GRDs con sus probabilidades:')
for grd, prob in prob_grds_sorted[:3]:
    print(f'  GRD {grd}: {prob * 100:.2f}%')
