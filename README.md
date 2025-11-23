# Iris Species Classification Project

Proyecto final del curso Data Mining. Clasifica especies del dataset Iris usando Random Forest y despliega una app en Streamlit.

## Contenido del repositorio
- Proyect.py  → aplicación principal en Streamlit
- requirements.txt → dependencias necesarias
- IRIS SPECIES CLASSIFICATION PROYECT.pdf → enunciado del proyecto
- README.md → documentación del proyecto

## Cómo ejecutar el proyecto (Windows)
1. Abrir PowerShell y moverse a la carpeta del proyecto:

   cd C:\Users\<tu_usuario>\Documents\iris_project

2. Crear y activar el entorno virtual:

   python -m venv venv  
   .\venv\Scripts\Activate

3. Instalar dependencias:

   pip install --upgrade pip  
   pip install -r requirements.txt

4. Ejecutar la app Streamlit:

   streamlit run Proyect.py

## Descripción del proyecto
Este proyecto utiliza el dataset Iris para entrenar un modelo de clasificación basado en Random Forest.  
El dashboard permite:

- Ver las métricas del modelo (Accuracy, Precision, Recall, F1).  
- Ingresar medidas de sépalo y pétalo para predecir la especie.  
- Visualizar la muestra nueva en un gráfico 3D.
- Revisar histogramas por especie.

## Dataset
Iris Dataset – UCI Machine Learning Repository  
https://www.kaggle.com/datasets/uciml/iris

## Video presentación
(Agrega aquí el link del video cuando lo subas)

## Autores
Eduardo Nolasco Gòmez

