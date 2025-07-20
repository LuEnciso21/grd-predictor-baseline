import pandas as pd
import numpy as np
import os
import re
import joblib
import pickle



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
    'TIPO GRD': ['M']
}




df = pd.DataFrame(data_cruda)

# --- Normalizar nombres columnas ---
df.columns = [col.upper().strip() for col in df.columns]

# --- Normalizaciones manuales para ciertos valores ---
normalizaciones = {
    'TIPO DE INGRESO': {
        'URGENCIAS': 'URGENCIA',
        'URGENCIA ADULTOS': 'URGENCIA',
        'U. ADULTOS': 'URGENCIA',
        'URGENCIA PEDIATRICA': 'URGENCIA',
        'URGENCIA OBSTETRICA': 'URGENCIA',
    },
    'SITUACION AL ALTA': {
        'MUERTO': 'FALLECIDO',
        'DEFUNCION': 'FALLECIDO',
    },
    'SERVICIOALTA': {
        'URGENCIAS': 'URGENCIA',
        'U. ADULTOS': 'URGENCIA',
    }
}

# --- Columnas categóricas a codificar ---
columnas_categoricas = [
    'ASEGURADORA -DESCRIPCION-',
    'GRUPO EDAD',
    'SEXO',
    'TIPO DE INGRESO',
    'CUIDADOS INTENSIVOS',
    'DX PRINCIPAL DE EGRESO',
    'DX PRINCIPAL DE EGRESO .1',
    'DX PPAL 3 CARACTERES',
    'DXR 1',
    'DXR 2',
    'DXR 3',
    'DXR 4',
    'DXR 5',
    'DXR-6',
    'SITUACION AL ALTA',
    'TIPO SERVICIO',
    'INFECCIONES',
    'INFECCION QUIRURGICA',
    'TIPO GRD'
]

# --- Aplicar limpieza de texto y normalizaciones ---
def limpiar_texto(valor):
    if isinstance(valor, str):
        valor = valor.strip().upper()
        return valor
    return valor

for col in df.columns:
    df[col] = df[col].map(limpiar_texto)

for col, mapa in normalizaciones.items():
    if col in df.columns:
        df[col] = df[col].map(lambda x: mapa.get(x, x))

# --- Mapeo columna a archivo encoder ---
mapa_columna_a_encoder = {
    'ASEGURADORA -DESCRIPCION-': 'encoder_ASEGURADORA_-DESCRIPCION-.pkl',
    'CUIDADOS INTENSIVOS': 'encoder_CUIDADOS_INTENSIVOS.pkl',
    'DXR-6': 'encoder_DXR-6.pkl',
    'DXR 1': 'encoder_DXR_1.pkl',
    'DXR 2': 'encoder_DXR_2.pkl',
    'DXR 3': 'encoder_DXR_3.pkl',
    'DXR 4': 'encoder_DXR_4.pkl',
    'DXR 5': 'encoder_DXR_5.pkl',
    'DX PPAL 3 CARACTERES': 'encoder_DX_PPAL_3_CARACTERES.pkl',
    'DX PRINCIPAL DE EGRESO': 'encoder_DX_PRINCIPAL_DE_EGRESO_.pkl',
    'DX PRINCIPAL DE EGRESO .1': 'encoder_DX_PRINCIPAL_DE_EGRESO_.1.pkl',
    'GRUPO EDAD': 'encoder_GRUPO_EDAD.pkl',
    'INFECCIONES': 'encoder_INFECCIONES.pkl',
    'INFECCION QUIRURGICA': 'encoder_INFECCION_QUIRURGICA.pkl',
    'SEXO': 'encoder_SEXO.pkl',
    'SITUACION AL ALTA': 'encoder_SITUACION_AL_ALTA.pkl',
    'TIPO DE INGRESO': 'encoder_TIPO_DE_INGRESO.pkl',
    'TIPO GRD': 'encoder_TIPO_GRD.pkl',
    'TIPO SERVICIO': 'encoder_TIPO_SERVICIO.pkl'
}

# --- Cargar y aplicar encoders ---
encoders_dir = "encoders"

df[columnas_categoricas] = df[columnas_categoricas].fillna('0').replace({0: '0'})

for col in columnas_categoricas:
    encoder_file = mapa_columna_a_encoder.get(col)
    if encoder_file:
        encoder_path = os.path.join(encoders_dir, encoder_file)
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            def transformar_valor(x):
                if x in encoder.classes_:
                    return encoder.transform([x])[0]
                else:
                    print(f"[WARN] Valor no visto en encoder para columna '{col}': '{x}', se asigna -1")
                    return -1
            df[col] = df[col].map(transformar_valor)
        else:
            print(f"[WARN] Archivo encoder no existe: {encoder_path}")
    else:
        print(f"[WARN] No se tiene encoder asignado para la columna: {col}")

# --- Procesar columnas numéricas: convertir coma a punto y a float ---
for col in ['PROC1', 'PROC2', 'PROC3']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# --- Seleccionar columnas esperadas para el modelo (ajusta según tu modelo) ---
columnas_esperadas = [
    'NUM CASO',
    'ASEGURADORA -CODIGO-',
    'ASEGURADORA -DESCRIPCION-',
    'EDAD',
    'GRUPO EDAD',
    'SEXO',
    'CODIGO DE CIUDAD',
    'FECHA DE INGRESO',
    'TIPO DE INGRESO',
    'FECHA DE EGRESO',
    'DIAS ESTANCIA',
    'SERVICIOALTA',
    'CUIDADOS INTENSIVOS',
    'DIAS DE UNIDAD CUIDADO INTENSIVO',
    'DX DE INGRESO',
    'DX PRINCIPAL DE EGRESO',
    'DX PRINCIPAL DE EGRESO .1',
    'DX PPAL 3 CARACTERES',
    'DXR 1',
    'DXR 2',
    'DXR 3',
    'DXR 4',
    'DXR 5',
    'DXR-6',
    'SITUACION AL ALTA',
    'PROC1',
    'PROC2',
    'PROC3',
    'TIPO SERVICIO',
    'INFECCIONES',
    'INFECCION QUIRURGICA',
    'TIPO GRD'
]

# --- Verificar que todas las columnas esperadas estén presentes ---
faltantes = [col for col in columnas_esperadas if col not in df.columns]
if faltantes:
    print(f"[ERROR] Faltan columnas esperadas: {faltantes}")
    exit()

# --- Ordenar dataframe con columnas esperadas ---
X = df[columnas_esperadas]

# --- Cargar modelo ---
modelo_path = "modelo_grd.pkl"
if not os.path.exists(modelo_path):
    print(f"[ERROR] No se encontró el archivo modelo: {modelo_path}")
    exit()

with open(modelo_path, "rb") as f:
    modelo = joblib.load('modelo_grd.pkl')

X = X.fillna(0)
# Predicciones probabilísticas
probs = modelo.predict_proba(X)[0]

# Top 3 índices y valores
top3_idx = np.argsort(probs)[::-1][:3]
top3_probs = probs[top3_idx]
top3_classes = modelo.classes_[top3_idx]


# --- Predecir ---
try:
    pred = modelo.predict(X)
    print(f"Predicción GRD: {pred[0]}")
except Exception as e:
    print(f"[ERROR] Al predecir: {e}")

print("\nTop 3 predicciones GRD con probabilidades:")
for i in range(3):
    print(f"{i+1}. GRD: {top3_classes[i]} con probabilidad {top3_probs[i]:.4f}")


print("Datos de entrada para la predicción:")
print(X.to_dict(orient='records')[0])