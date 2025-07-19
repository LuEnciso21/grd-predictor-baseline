import joblib
import pandas as pd
import os
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

# Rutas a los archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, 'modelo_grd.pkl')
COLUMNAS_PATH = os.path.join(BASE_DIR, 'columnas_modelo.pkl')

# Cargar modelo y columnas
modelo = joblib.load(MODELO_PATH)
columnas = joblib.load(COLUMNAS_PATH)

def alinear_columnas(df, columnas_esperadas):
    """Alinea las columnas del DataFrame con las columnas esperadas por el modelo"""
    # Asegurar que todas las columnas esperadas estén presentes, rellenando con 0 si no
    for col in columnas_esperadas:
        if col not in df.columns:
            df[col] = 0
    # Asegurar el orden de las columnas
    df = df[columnas_esperadas]
    return df

# Datos de prueba - Datos ya codificados y preprocesados
data = {
    'NUM CASO': [0.9972975383732675],
    'ASEGURADORA -CODIGO-': [0.6975308641975309],
    'ASEGURADORA -DESCRIPCION-': [0.9999999999999999],
    'EDAD': [0.9174311926605505],
    'GRUPO EDAD': [0.3940298507462687],
    'SEXO': [10.0],
    'CODIGO DE CIUDAD': [0.0],
    'FECHA DE INGRESO': [0.798405871750444],
    'TIPO DE INGRESO': [0.0],
    'FECHA DE EGRESO': [0.9312520757223512],
    'DIAS ESTANCIA': [0.02298850574712644],
    'SERVICIOALTA': [0.1227436823104693],
    'CUIDADOS INTENSIVOS': [0.0],
    'DIAS DE UNIDAD CUIDADO INTENSIVO': [0.0],
    'DX DE INGRESO': [1.0],
    'DX PRINCIPAL DE EGRESO': [0.015706806282722512],
    'DX PRINCIPAL DE EGRESO .1': [0.01824817518248175],
    'DX PPAL 3 CARACTERES': [0.01824817518248175],
    'DXR 1': [0.010174418604651162],
    'DXR 2': [0.008161044613710554],
    'DXR 3': [0.020072992700729927],
    'DXR 4': [0.0800711743772242],
    'DXR 5': [0.6485714285714285],
    'DXR-6': [0.5274102079395085],
    'SITUACION AL ALTA': [0.0],
    'PROC1': [0.45332262597871903],
    'PROC2': [0.8904276985743382],
    'PROC3': [0.8896953588223268],
    'TIPO SERVICIO': [1.0],
    'INFECCIONES': [0.0],
    'INFECCION QUIRURGICA': [0.0]
}

df_test = pd.DataFrame(data)

# Alinea las columnas del DataFrame de prueba directamente con las columnas esperadas por el modelo
df_final = alinear_columnas(df_test, columnas)

# Realizar la predicción
prediccion = modelo.predict(df_final)[0]
probabilidades = modelo.predict_proba(df_final)[0]

# Obtener los GRDs y sus probabilidades
grds = modelo.classes_
prob_grds = list(zip(grds, probabilidades))

# Ordenar por probabilidad de mayor a menor
prob_grds_sorted = sorted(prob_grds, key=lambda x: x[1], reverse=True)

# Mostrar el GRD predicho y las 3 principales probabilidades
print(f'GRD Predicho: {prediccion}')
print('Top 3 GRDs con sus probabilidades:')
for grd, prob in prob_grds_sorted[:3]:
    print(f'  GRD {grd}: {prob * 100:.2f}%') 