# GRD Baseline Predictor

*IA para Clasificación de Grupos Relacionados por el Diagnóstico con Despliegue Web*

---

## Descripción

Este proyecto implementa un modelo de inteligencia artificial para predecir los **Grupos Relacionados por el Diagnóstico (GRD)** más probables a partir de datos clínicos crudos. El sistema incluye un pipeline completo de preprocesamiento de datos, entrenamiento y evaluación del modelo, y una interfaz web para realizar predicciones en tiempo real.

El objetivo es facilitar la clasificación automática de pacientes en grupos diagnósticos para optimizar la gestión hospitalaria y la facturación.

---

## Instalación

1. Clona el repositorio:

```bash

git clone https://github.com/LuEnciso21/grd-baseline-predictor.git

```
2. Configura Django:

```bash
pip install django
python manage.py migrate
python manage.py runserver

```

## Entrenamiento del modelo

El notebook para entrenamiento se llama `PFInfoMed.ipynb`. Inlcuye:

- Limpieza y preprocesamiento de datos originales.
- Selección y codificación de variables relevantes.
- Entrenamiento y ajuste de hiperparámetros con GridSearchCV.
- Evaluación y métricas (accuracy, F1-score, etc.).

---

## Uso

### Predicción desde código Python

En el archivo `PROBANDO2.py` :

Puede hacer la predicción de un sujeto a partir de un diccionario con los datos crudos.

---

## Tecnologías utilizadas

- Python 
- Scikit-learn (RandomForestClassifier)
- Pandas, NumPy
- Django para backend y despliegue web
- Joblib para serialización de modelos
- HTML/CSS para interfaz sencilla

---

## Contacto

Luisa Enciso

Maria Ostos

Santiago Rivera
