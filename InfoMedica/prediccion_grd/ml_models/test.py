import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, 'modelo_grd.pkl')
COLUMNAS_PATH = os.path.join(BASE_DIR, 'columnas_modelo.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler_modelo.pkl')

scaler = joblib.load(SCALER_PATH)
modelo = joblib.load(MODELO_PATH)
columnas = joblib.load(COLUMNAS_PATH)

def codificar_variables(df):
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns

    for col in columnas_categoricas:
        valores_unicos = df[col].dropna().unique()
        if len(valores_unicos) == 2:
            df[col] = df[col].map({valores_unicos[0]: 0, valores_unicos[1]: 1})
        else:
            frecuencia = df[col].value_counts()
            df[col] = df[col].map(frecuencia)
    return df

def escalar_datos(df, columnas_a_escalar, scaler):
    df_scaled = df.copy()
    df_scaled[columnas_a_escalar] = scaler.transform(df_scaled[columnas_a_escalar])
    return df_scaled

def alinear_columnas(df, columnas_esperadas):
    df = df.copy()
    # Limpiar columnas del df
    df.columns = df.columns.str.strip()
    
    # Limpiar columnas esperadas
    columnas_esperadas = [col.strip() for col in columnas_esperadas]

    # Mostrar diferencias
    set_df = set(df.columns)
    set_esp = set(columnas_esperadas)
    print("[INFO] Columnas en df pero no esperadas:", set_df - set_esp)
    print("[INFO] Columnas esperadas pero no en df:", set_esp - set_df)

    for col in columnas_esperadas:
        if col not in df.columns:
            print(f"[WARN] Columna faltante: {col}")
            df[col] = 0
    return df[columnas_esperadas]

def preparar_datos_para_modelo(df_input):
    df = df_input.copy()

    # Codificar variables categóricas
    df = codificar_variables(df)

    # Rellenar NaNs
    df.fillna(0, inplace=True)

    # Agregar epsilon
    epsilon = 1e-3
    df.iloc[:, :] += epsilon

    # Alinear columnas primero
    df_alineado = alinear_columnas(df, columnas)

    # Escalar solo las columnas numéricas en el orden correcto
    columnas_a_escalar = df_alineado.select_dtypes(include=['int64', 'float64']).columns
    df_scaled = df_alineado.copy()
    df_scaled[columnas_a_escalar] = scaler.transform(df_alineado[columnas_a_escalar])

    return df_scaled

import pandas as pd

# Datos crudos transformados en formato correcto
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

import pandas as pd

data_cruda_3 = {
    'NUM CASO': [413799],
    'ASEGURADORA -CODIGO-': ['204'],
    'ASEGURADORA -DESCRIPCION-': ['Aseguradora_29'],
    'EDAD': [38],
    'GRUPO EDAD': ['35-39'],
    'SEXO': ['F'],
    'CODIGO DE CIUDAD': ['5001'],
    'FECHA DE INGRESO': ['30052005'],
    'TIPO DE INGRESO': ['URGENCIA'],
    'FECHA DE EGRESO': ['19062005'],
    'DIAS ESTANCIA': [20],
    'SERVICIOALTA': ['29'],
    'CUIDADOS INTENSIVOS': ['NO'],
    'DIAS DE UNIDAD CUIDADO INTENSIVO': [0],
    'DX DE INGRESO': ['2'],
    'DX PRINCIPAL DE EGRESO': ['I635'],
    'DX PRINCIPAL DE EGRESO .1': ['I63'],
    'DX PPAL 3 CARACTERES': ['INFARTO CEREBRAL'],
    'DXR 1': ['I652'],
    'DXR 2': ['C189'],
    'DXR 3': ['C787'],
    'DXR 4': ['C788'],
    'DXR 5': ['N390'],
    'DXR-6': ['A048'],
    'SITUACION AL ALTA': ['ALTA MEDICA'],
    'PROC1': ['43.11'],
    'PROC2': ['54.91'],
    'PROC3': ['87.44'],
    'TIPO SERVICIO': ['URGENCIA ADULTOS'],
    'INFECCIONES': ['NO'],
    'INFECCION QUIRURGICA': ['NO'],
    'TIPO GRD':['M']
}

data_cruda_2 = {
    'NUM CASO': [411289],
    'ASEGURADORA -CODIGO-': ['313'],
    'ASEGURADORA -DESCRIPCION-': ['Aseguradora_56'],
    'EDAD': [79],
    'GRUPO EDAD': ['75-79'],
    'SEXO': ['M'],
    'CODIGO DE CIUDAD': ['5001'],
    'FECHA DE INGRESO': ['10012005'],
    'TIPO DE INGRESO': ['URGENCIA'],
    'FECHA DE EGRESO': ['20012005'],
    'DIAS ESTANCIA': [10],
    'SERVICIOALTA': ['38'],
    'CUIDADOS INTENSIVOS': ['SI'],
    'DIAS DE UNIDAD CUIDADO INTENSIVO': [2],
    'DX DE INGRESO': ['2'],
    'DX PRINCIPAL DE EGRESO': ['I132'],
    'DX PRINCIPAL DE EGRESO .1': ['I13'],
    'DX PPAL 3 CARACTERES': ['ENFERMEDAD CARDIORRENAL HIPERTENSIVA'],
    'DXR 1': ['I48'],
    'DXR 2': ['J158'],
    'DXR 3': ['Y95'],
    'DXR 4': ['J441'],
    'DXR 5': ['N390'],
    'DXR-6': ['A099'],
    'SITUACION AL ALTA': ['FALLECIDO'],
    'PROC1': ['87.44'],
    'PROC2': ['87.03'],
    'PROC3': ['88.72'],
    'TIPO SERVICIO': ['URGENCIA ADULTOS'],
    'INFECCIONES': ['NO'],
    'INFECCION QUIRURGICA': ['NO'],
    'TIPO GRD':['M']
}

df_input_2 = pd.DataFrame(data_cruda_2)



# Reemplazar comas si vienen mal, y convertir a float
for col in df_input_2.columns:
    if 'PROC' in col and df_input_2[col].notnull().any():
        df_input_2[col] = df_input_2[col].astype(str).str.replace(',', '.').astype(float)
df_preparado_2 = preparar_datos_para_modelo(df_input_2)

# Predecir GRD
prediccion_2 = modelo.predict(df_preparado_2)[0]
probabilidades_2 = modelo.predict_proba(df_preparado_2)[0]
grds_2 = modelo.classes_
prob_grds_sorted_2 = sorted(zip(grds_2, probabilidades_2), key=lambda x: x[1], reverse=True)

# Mostrar resultados
print(f'GRD Predicho2: {prediccion_2}')
print('Top 3 GRDs con sus probabilidades:')
for grd, prob in prob_grds_sorted_2[:3]:
    print(f'  GRD {grd}: {prob * 100:.2f}%')

df_input = pd.DataFrame(data_cruda)
df_preparado = preparar_datos_para_modelo(df_input)

# Predicción
prediccion = modelo.predict(df_preparado)[0]
probabilidades = modelo.predict_proba(df_preparado)[0]
grds = modelo.classes_
prob_grds_sorted = sorted(zip(grds, probabilidades), key=lambda x: x[1], reverse=True)

# Mostrar
print(f'GRD Predicho1: {prediccion}')
print('Top 3 GRDs con sus probabilidades:')
for grd, prob in prob_grds_sorted[:3]:
    print(f'  GRD {grd}: {prob * 100:.2f}%')
print(np.allclose(df_preparado, df_preparado_2))
print(df_preparado_2.head().T)
