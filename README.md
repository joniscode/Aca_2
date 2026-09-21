# ACA 2 - Titanic con Red Neuronal y Streamlit

Aplicación interactiva desarrollada para el ACA 2 de Aprendizaje Automático.

## Archivos requeridos

El repositorio debe contener:

- `app.py`
- `modelo_titanic.keras`
- `scaler.pkl`
- `metadata.json`
- `requirements.txt`

Los archivos `modelo_titanic.keras` y `scaler.pkl` se generaron al ejecutar el notebook
`ACA2_Titanic.ipynb` en Google Colab.

## Descargar los artefactos desde Colab

Después de ejecutar todo el notebook, se ejecuta una celda nueva donde se generan estos nuevos archivos:

```python
from google.colab import files

files.download("modelo_titanic.keras")
files.download("scaler.pkl")
files.download("metadata.json")
```

luego Guarde estos tres archivos en mi computador para poder subirlos a streamlit.

## Estructura final del repositorio

```text
ACA_2/
├── app.py
├── modelo_titanic.keras
├── scaler.pkl
├── metadata.json
├── requirements.txt
└── README.md
```

## Probar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

