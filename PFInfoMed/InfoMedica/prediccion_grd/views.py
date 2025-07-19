from django.shortcuts import render
import joblib
import pandas as pd
from django import forms
import os
from sklearn.preprocessing import MinMaxScaler
import unicodedata

# Rutas absolutas a los archivos del modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, 'ml_models', 'modelo_grd.pkl')
COLUMNAS_PATH = os.path.join(BASE_DIR, 'ml_models', 'columnas_modelo.pkl')

# Cargar modelo y columnas
modelo = joblib.load(MODELO_PATH)
columnas = joblib.load(COLUMNAS_PATH)

def preprocesar_datos_completo(df):
    """Preprocesamiento completo que coincide con el entrenamiento"""
    df_copy = df.copy()
    
    # Limpiar texto
    def limpiar_texto(texto):
        if isinstance(texto, str):
            texto = texto.upper()
            texto = unicodedata.normalize('NFKD', texto)
            texto = ''.join([c for c in texto if not unicodedata.combining(c)])
            return texto
        return texto

    # Aplicar solo a columnas de texto
    columnas_texto = df_copy.select_dtypes(include='object').columns
    df_copy[columnas_texto] = df_copy[columnas_texto].applymap(limpiar_texto)

    # Limpiar nombres de columnas
    df_copy.columns = [limpiar_texto(col) for col in df_copy.columns]

    # Normalización de valores equivalentes
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
            'NO': 'NO APLICA',
        },
        'TIPO SERVICIO': {
            'URGENCIA ADULTOS': 'URGENCIA ADULTOS',
            'URGENCIAS PEDIATRICAS': 'URGENCIAS PEDIATRICAS',
            'NO': 'NO APLICA',
        }
    }

    for columna, reemplazos in normalizaciones.items():
        if columna in df_copy.columns:
            df_copy[columna] = df_copy[columna].replace(reemplazos)

    # Codificación por frecuencia
    columnas_categoricas = df_copy.select_dtypes(include=['object', 'category']).columns

    for col in columnas_categoricas:
        valores_unicos = df_copy[col].dropna().unique()
        if len(valores_unicos) == 2:
            # Para variables binarias, asegurar que "NO" sea 0
            if 'NO' in valores_unicos:
                no_valor = 'NO'
                otro_valor = [v for v in valores_unicos if v != 'NO'][0]
                df_copy[col] = df_copy[col].map({no_valor: 0, otro_valor: 1})
            else:
                df_copy[col] = df_copy[col].map({valores_unicos[0]: 0, valores_unicos[1]: 1})
        else:
            frecuencia = df_copy[col].value_counts()
            df_copy[col] = df_copy[col].map(frecuencia)

    # Normalización Min-Max
    df_clean = df_copy.fillna(0)
    epsilon = 1e-3
    df_adjusted = df_clean.copy()
    if df_adjusted.shape[1] > 1:
        df_adjusted.iloc[:, :-1] += epsilon
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df_adjusted.iloc[:, :-1])
        df_f = pd.DataFrame(scaled_values, columns=df_adjusted.columns[:-1])
        df_f[df_adjusted.columns[-1]] = df_adjusted[df_adjusted.columns[-1]]
    else:
        df_f = df_adjusted.copy()

    # One-hot encoding
    df_onehot = pd.get_dummies(df_f, drop_first=True)

    # Alinear columnas con el modelo
    for col in columnas:
        if col not in df_onehot:
            df_onehot[col] = 0
    
    df_final = df_onehot[columnas]
    
    return df_final

# Generar dinámicamente el formulario con TODAS las columnas del modelo
class DynamicPrediccionGRDForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Crear campos para todas las columnas del modelo
        for col in columnas:
            # Determinar el tipo de campo basado en el nombre de la columna
            if 'EDAD' in col.upper():
                self.fields[col] = forms.IntegerField(
                    label=col, 
                    required=False,
                    widget=forms.NumberInput(attrs={'class': 'form-control'})
                )
            elif 'SEXO' in col.upper():
                self.fields[col] = forms.ChoiceField(
                    label=col, 
                    choices=[('', 'Seleccionar...'), ('M', 'Masculino'), ('F', 'Femenino')], 
                    required=False,
                    widget=forms.Select(attrs={'class': 'form-control'})
                )
            elif 'DIAS' in col.upper() or 'ESTANCIA' in col.upper():
                self.fields[col] = forms.IntegerField(
                    label=col, 
                    required=False,
                    widget=forms.NumberInput(attrs={'class': 'form-control'})
                )
            elif 'CUIDADOS INTENSIVOS' in col.upper() or 'INFECCION' in col.upper():
                self.fields[col] = forms.ChoiceField(
                    label=col, 
                    choices=[('', 'Seleccionar...'), ('NO', 'No'), ('SI', 'Sí')], 
                    required=False,
                    widget=forms.Select(attrs={'class': 'form-control'})
                )
            elif 'TIPO DE INGRESO' in col.upper():
                self.fields[col] = forms.ChoiceField(
                    label=col, 
                    choices=[('', 'Seleccionar...'), ('URGENCIA', 'Urgencia'), ('PROGRAMADO', 'Programado')], 
                    required=False,
                    widget=forms.Select(attrs={'class': 'form-control'})
                )
            elif 'SITUACION AL ALTA' in col.upper():
                self.fields[col] = forms.ChoiceField(
                    label=col, 
                    choices=[('', 'Seleccionar...'), ('ALTA MÉDICA', 'Alta Médica'), ('FALLECIDO', 'Fallecido')], 
                    required=False,
                    widget=forms.Select(attrs={'class': 'form-control'})
                )
            elif 'SERVICIOALTA' in col.upper():
                self.fields[col] = forms.ChoiceField(
                    label=col, 
                    choices=[('', 'Seleccionar...'), ('URGENCIA', 'Urgencia'), ('NO', 'NO aplica')], 
                    required=False,
                    widget=forms.Select(attrs={'class': 'form-control'})
                )
            elif 'TIPO SERVICIO' in col.upper():
                self.fields[col] = forms.ChoiceField(
                    label=col, 
                    choices=[('', 'Seleccionar...'), ('URGENCIA ADULTOS', 'Urgencia Adultos'), ('URGENCIAS PEDIATRICAS', 'Urgencias Pediátricas'), ('NO', 'NO aplica')], 
                    required=False,
                    widget=forms.Select(attrs={'class': 'form-control'})
                )
            elif 'CODIGO' in col.upper() or 'NUM' in col.upper():
                self.fields[col] = forms.IntegerField(
                    label=col, 
                    required=False,
                    widget=forms.NumberInput(attrs={'class': 'form-control'})
                )
            else:
                # Campo de texto para el resto
                self.fields[col] = forms.CharField(
                    label=col, 
                    required=False,
                    widget=forms.TextInput(attrs={'class': 'form-control'})
                )

def predecir_grd(request):
    resultado = None
    probabilidades = None
    df_html = None
    df_original_html = None
    
    if request.method == 'POST':
        form = DynamicPrediccionGRDForm(request.POST)
        if form.is_valid():
            datos = form.cleaned_data
            
            # Crear diccionario con todos los datos del formulario
            data = {}
            for k, v in datos.items():
                if v in [None, '']:
                    data[k] = float('nan')
                else:
                    try:
                        if '.' in str(v):
                            data[k] = float(v)
                        else:
                            data[k] = int(v)
                    except Exception:
                        data[k] = str(v)
            
            # Rellenar los campos faltantes con NaN (como en el test)
            for col in columnas:
                if col not in data:
                    data[col] = float('nan')

            X_nuevo = pd.DataFrame([data])
            df_original_html = X_nuevo.to_html(classes='table table-bordered table-sm', index=False)

            # Aplicar preprocesamiento completo (igual que en el test)
            X_procesado = preprocesar_datos_completo(X_nuevo)

            df_html = X_procesado.to_html(classes='table table-bordered table-sm', index=False)
            
            # Obtener predicción y probabilidades
            prediccion = modelo.predict(X_procesado)[0]
            probabilidades_completas = modelo.predict_proba(X_procesado)[0]
            
            # Obtener las clases (GRDs) del modelo
            clases_modelo = modelo.classes_
            
            # Crear lista de tuplas (GRD, probabilidad) y ordenar por probabilidad
            grd_probabilidades = list(zip(clases_modelo, probabilidades_completas))
            grd_probabilidades.sort(key=lambda x: x[1], reverse=True)
            
            # Tomar los 3 primeros y convertir a porcentajes (x100)
            resultado = prediccion
            probabilidades = [(grd, prob * 100) for grd, prob in grd_probabilidades[:3]]
            
            return render(request, 'prediccion_grd/formulario.html', {
                'form': form, 
                'resultado': resultado, 
                'probabilidades': probabilidades,
                'df_html': df_html, 
                'df_original_html': df_original_html
            })
    else:
        form = DynamicPrediccionGRDForm()
    
    return render(request, 'prediccion_grd/formulario.html', {
        'form': form, 
        'resultado': resultado, 
        'probabilidades': probabilidades,
        'df_html': df_html, 
        'df_original_html': df_original_html
    })