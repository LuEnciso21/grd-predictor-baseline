import os
import joblib
import pandas as pd
import numpy as np
from django import forms
from django.shortcuts import render

# Ruta base del archivo actual (prediccion_grd/views.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de los modelos
COLUMNAS_PATH = os.path.join(BASE_DIR, '..', '..', 'columnas_modelo.pkl')
MODELO_PATH = os.path.join(BASE_DIR, '..', '..', 'modelo_grd.pkl')
ENCODERS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'encoders'))


# Cargar columnas del modelo
columnas_modelo = joblib.load(COLUMNAS_PATH)

# Columnas que no deben mostrarse en el formulario pero deben enviarse como None al modelo
columnas_ocultas = [
    'CUIDADOS INTENSIVOS',
    'INFECCIONES',
    'INFECCION QUIRURGICA',
    'DX DE INGRESO',
    'DIAS DE UNIDAD CUIDADO INTENSIVO',
    'SITUACION AL ALTA'
]



class DynamicPrediccionGRDForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for col in columnas_modelo:
            if col in columnas_ocultas:
                continue
            if col == 'ASEGURADORA -DESCRIPCION-':
                self.fields[col] = forms.CharField(label=col, required=False, widget=forms.TextInput(attrs={'readonly': 'readonly'}))
            elif 'EDAD' in col.upper():
                self.fields[col] = forms.IntegerField(label=col, required=False)
            elif 'SEXO' in col.upper():
                self.fields[col] = forms.ChoiceField(label=col, choices=[('', 'Seleccionar...'), ('M', 'Masculino'), ('F', 'Femenino')], required=False)
            elif 'DIAS' in col.upper() or 'ESTANCIA' in col.upper():
                self.fields[col] = forms.IntegerField(label=col, required=False)
            else:
                self.fields[col] = forms.CharField(label=col, required=False)

def predecir_grd(request):
    resultado = None
    probabilidades = None
    datos_procesados = None
    advertencias = []

    if request.method == 'POST':
        form = DynamicPrediccionGRDForm(request.POST)
        if form.is_valid():
            data_cruda = {}
            for k, v in form.cleaned_data.items():
                if v in [None, '']:
                    data_cruda[k] = [float('nan')]
                else:
                    try:
                        if '.' in str(v):
                            data_cruda[k] = [float(v)]
                        else:
                            data_cruda[k] = [int(v)]
                    except Exception:
                        data_cruda[k] = [str(v)]

            for col in columnas_ocultas:
                data_cruda[col] = [None]

            df = pd.DataFrame(data_cruda)
            df.columns = [col.upper().strip() for col in df.columns]

        
            def limpiar_texto(valor):
                if isinstance(valor, str):
                    return valor.strip().upper()
                return valor

            for col in df.columns:
                df[col] = df[col].map(limpiar_texto)

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

            for col, mapa in normalizaciones.items():
                if col in df.columns:
                    df[col] = df[col].map(lambda x: mapa.get(x, x))

            columnas_categoricas = [
                'ASEGURADORA -DESCRIPCION-','SEXO', 'TIPO DE INGRESO',
                'CUIDADOS INTENSIVOS', 'DX PRINCIPAL DE EGRESO', 'DX PRINCIPAL DE EGRESO .1',
                'DX PPAL 3 CARACTERES', 'DXR 1', 'DXR 2', 'DXR 3', 'DXR 4', 'DXR 5', 'DXR-6',
                'SITUACION AL ALTA', 'TIPO SERVICIO', 'INFECCIONES', 'INFECCION QUIRURGICA',
                'TIPO GRD'
            ]

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

            for col in columnas_categoricas:
                if col in df.columns:
                    encoder_file = mapa_columna_a_encoder.get(col)
                    if encoder_file:
                        encoder_path = os.path.join(ENCODERS_DIR, encoder_file)
                        if os.path.exists(encoder_path):
                            encoder = joblib.load(encoder_path)
                            def transformar_valor(x):
                                if pd.isna(x):
                                    return None
                                if x in encoder.classes_:
                                    return encoder.transform([x])[0]
                                else:
                                    advertencias.append(f"Valor desconocido en '{col}': '{x}' (asignado -1)")
                                    return -1
                            df[col] = df[col].map(transformar_valor)
                        else:
                            advertencias.append(f"Falta encoder para columna: {col}")
                    else:
                        advertencias.append(f"No hay encoder asignado para: {col}")

            for col in ['PROC1', 'PROC2', 'PROC3']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            columnas_esperadas = columnas_modelo
            faltantes = [col for col in columnas_esperadas if col not in df.columns]
            if faltantes:
                advertencias.append(f"Faltan columnas esperadas: {faltantes}")
                X = df
            else:
                X = df[columnas_esperadas]

            if not os.path.exists(MODELO_PATH):
                advertencias.append("Archivo de modelo no encontrado.")
                return render(request, 'prediccion_grd/formulario.html', {
                    'form': form, 'advertencias': advertencias
                })

            modelo = joblib.load(MODELO_PATH)
            X = X.fillna(0)

            try:
                probs = modelo.predict_proba(X)[0]
                pred = modelo.predict(X)

                top3_idx = np.argsort(probs)[::-1][:3]
                top3_probs = probs[top3_idx]
                top3_classes = modelo.classes_[top3_idx]

                resultado = pred[0]
                probabilidades = list(zip(top3_classes, top3_probs * 100))
                datos_procesados = X.to_dict(orient='records')[0]

            except Exception as e:
                advertencias.append(f"Error al predecir: {e}")

            return render(request, 'prediccion_grd/formulario.html', {
                'form': form,
                'resultado': resultado,
                'probabilidades': probabilidades,
                'datos_dict': datos_procesados,
                'advertencias': advertencias
            })

    else:
        form = DynamicPrediccionGRDForm()

    return render(request, 'prediccion_grd/formulario.html', {
        'form': form,
        'resultado': None,
        'probabilidades': None,
        'datos_dict': None,
        'advertencias': []
    })
import os
import joblib
import pandas as pd
import numpy as np
from django import forms
from django.shortcuts import render

# Ruta base del archivo actual (prediccion_grd/views.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de los modelos
COLUMNAS_PATH = os.path.join(BASE_DIR, 'ml_models', 'columnas_modelo.pkl')
MODELO_PATH = os.path.join(BASE_DIR, 'ml_models', 'modelo_grd.pkl')
ENCODERS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'encoders'))

# Ruta al Excel con las aseguradoras
EXCEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'DataBase.xlsx'))

# Cargar columnas del modelo
columnas_modelo = joblib.load(COLUMNAS_PATH)

# Columnas que no deben mostrarse en el formulario pero deben enviarse como None al modelo
columnas_ocultas = [
    'CUIDADOS INTENSIVOS',
    'INFECCIONES',
    'INFECCION QUIRURGICA',
    'DX DE INGRESO',
    'DIAS DE UNIDAD CUIDADO INTENSIVO',
    'SITUACION AL ALTA',
    'GRUPO EDAD',
    'ASEGURADORA -DESCRIPCION-'
]

class DynamicPrediccionGRDForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for col in columnas_modelo:
            if col in columnas_ocultas:
                continue
            if col == 'ASEGURADORA -DESCRIPCION-':
                self.fields[col] = forms.CharField(label=col, required=False, widget=forms.TextInput(attrs={'readonly': 'readonly'}))
            elif 'EDAD' in col.upper():
                self.fields[col] = forms.IntegerField(label=col, required=False)
            elif 'SEXO' in col.upper():
                self.fields[col] = forms.ChoiceField(label=col, choices=[('', 'Seleccionar...'), ('M', 'Masculino'), ('F', 'Femenino')], required=False)
            elif 'DIAS' in col.upper() or 'ESTANCIA' in col.upper():
                self.fields[col] = forms.IntegerField(label=col, required=False)
            else:
                self.fields[col] = forms.CharField(label=col, required=False)

def predecir_grd(request):
    resultado = None
    probabilidades = None
    datos_procesados = None
    advertencias = []

    if request.method == 'POST':
        form = DynamicPrediccionGRDForm(request.POST)
        if form.is_valid():
            data_cruda = {}
            for k, v in form.cleaned_data.items():
                if v in [None, '']:
                    data_cruda[k] = [float('nan')]
                else:
                    try:
                        if '.' in str(v):
                            data_cruda[k] = [float(v)]
                        else:
                            data_cruda[k] = [int(v)]
                    except Exception:
                        data_cruda[k] = [str(v)]

            # Agregar columnas ocultas como None
            for col in columnas_ocultas:
                data_cruda[col] = [None]

            df = pd.DataFrame(data_cruda)
            df.columns = [col.upper().strip() for col in df.columns]

            
            # Limpieza de texto
            def limpiar_texto(valor):
                if isinstance(valor, str):
                    return valor.strip().upper()
                return valor

            for col in df.columns:
                df[col] = df[col].map(limpiar_texto)

            # Normalizaciones manuales
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

            for col, mapa in normalizaciones.items():
                if col in df.columns:
                    df[col] = df[col].map(lambda x: mapa.get(x, x))

            columnas_categoricas = [
                'SEXO', 'TIPO DE INGRESO',
                'CUIDADOS INTENSIVOS', 'DX PRINCIPAL DE EGRESO', 'DX PRINCIPAL DE EGRESO .1',
                'DX PPAL 3 CARACTERES', 'DXR 1', 'DXR 2', 'DXR 3', 'DXR 4', 'DXR 5', 'DXR-6',
                'SITUACION AL ALTA', 'TIPO SERVICIO', 'INFECCIONES', 'INFECCION QUIRURGICA',
                'TIPO GRD'
            ]

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

            for col in columnas_categoricas:
                encoder_file = mapa_columna_a_encoder.get(col)
                if encoder_file:
                    encoder_path = os.path.join(ENCODERS_DIR, encoder_file)
                    if os.path.exists(encoder_path):
                        encoder = joblib.load(encoder_path)
                        def transformar_valor(x):
                            if pd.isna(x):
                                return None
                            if x in encoder.classes_:
                                return encoder.transform([x])[0]
                            else:
                                advertencias.append(f"Valor desconocido en '{col}': '{x}' (asignado -1)")
                                return -1
                        df[col] = df[col].map(transformar_valor)
                    else:
                        advertencias.append(f"Falta encoder para columna: {col}")
                else:
                    advertencias.append(f"No hay encoder asignado para: {col}")

            for col in ['PROC1', 'PROC2', 'PROC3']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            columnas_esperadas = columnas_modelo
            faltantes = [col for col in columnas_esperadas if col not in df.columns]
            if faltantes:
                advertencias.append(f"Faltan columnas esperadas: {faltantes}")
                X = df
            else:
                X = df[columnas_esperadas]

            if not os.path.exists(MODELO_PATH):
                advertencias.append("Archivo de modelo no encontrado.")
                return render(request, 'prediccion_grd/formulario.html', {
                    'form': form, 'advertencias': advertencias, 
                })

            modelo = joblib.load(MODELO_PATH)
            X = X.fillna(0)

            try:
                probs = modelo.predict_proba(X)[0]
                pred = modelo.predict(X)

                top3_idx = np.argsort(probs)[::-1][:3]
                top3_probs = probs[top3_idx]
                top3_classes = modelo.classes_[top3_idx]

                resultado = pred[0]
                probabilidades = list(zip(top3_classes, top3_probs * 100))
                datos_procesados = X.to_dict(orient='records')[0]

            except Exception as e:
                advertencias.append(f"Error al predecir: {e}")

            return render(request, 'prediccion_grd/formulario.html', {
                'form': form,
                'resultado': resultado,
                'probabilidades': probabilidades,
                'datos_dict': datos_procesados,
                'advertencias': advertencias,
            })

    else:
        form = DynamicPrediccionGRDForm()

    return render(request, 'prediccion_grd/formulario.html', {
        'form': form,
        'resultado': None,
        'probabilidades': None,
        'datos_dict': None,
        'advertencias': [],
    })