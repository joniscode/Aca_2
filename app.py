
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "modelo_titanic.keras"
SCALER_PATH = APP_DIR / "scaler.pkl"
METADATA_PATH = APP_DIR / "metadata.json"

st.set_page_config(
    page_title="Titanic | Red Neuronal",
    page_icon="🚢",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.18);
        margin-bottom: 1rem;
      }
      .prediction-card {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.2);
        text-align: center;
      }
      .small-note {opacity: .75; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def cargar_artefactos():
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, scaler, metadata

def artefactos_disponibles():
    faltantes = [
        p.name for p in [MODEL_PATH, SCALER_PATH, METADATA_PATH]
        if not p.exists()
    ]
    return faltantes

faltantes = artefactos_disponibles()
if faltantes:
    st.error(
        "Faltan archivos necesarios para ejecutar la aplicación: "
        + ", ".join(faltantes)
    )
    st.info(
        "Copia estos archivos desde Google Colab al mismo directorio de app.py "
        "y vuelve a desplegar la aplicación."
    )
    st.stop()

modelo, scaler, metadata = cargar_artefactos()

features = metadata["features"]
metricas = metadata["metricas"]

st.markdown(
    """
    <div class="hero">
      <h1>🚢 Predictor de supervivencia del Titanic</h1>
      <p>
        Aplicación interactiva del ACA 2 de Aprendizaje Automático.
        La predicción se realiza con una red neuronal multicapa entrenada
        sobre el mismo conjunto de datos utilizado en el ACA 1.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_metricas, tab_comp, tab_modelo = st.tabs(
    ["🔮 Predicción", "📊 Métricas", "📈 Comparación", "🧠 Modelo"]
)

with tab_pred:
    st.subheader("Datos del pasajero")
    st.caption(
        "Ingresa las características del pasajero. "
        "FamilySize e IsAlone se calculan automáticamente."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        pclass = st.selectbox(
            "Clase del pasajero",
            options=[1, 2, 3],
            format_func=lambda x: f"{x}ª clase"
        )
        sexo = st.selectbox(
            "Sexo",
            options=["female", "male"],
            format_func=lambda x: "Femenino" if x == "female" else "Masculino"
        )
        edad = st.slider("Edad", 0.0, 80.0, 30.0, 1.0)

    with c2:
        tarifa = st.number_input(
            "Tarifa pagada",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0
        )
        sibsp = st.number_input(
            "Hermanos / cónyuges a bordo (SibSp)",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
        parch = st.number_input(
            "Padres / hijos a bordo (Parch)",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

    with c3:
        embarked = st.selectbox(
            "Puerto de embarque",
            options=["S", "C", "Q"],
            format_func=lambda x: {
                "S": "Southampton (S)",
                "C": "Cherbourg (C)",
                "Q": "Queenstown (Q)"
            }[x]
        )

        family_size = int(sibsp + parch + 1)
        is_alone = 1 if family_size == 1 else 0

        st.metric("Tamaño familiar", family_size)
        st.metric("¿Viaja solo?", "Sí" if is_alone else "No")

    if st.button("Realizar predicción", type="primary", use_container_width=True):
        sex_male = 1 if sexo == "male" else 0
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0

        entrada = {
            "Pclass": pclass,
            "Age": edad,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": tarifa,
            "FamilySize": family_size,
            "IsAlone": is_alone,
            "Sex_male": sex_male,
            "Embarked_Q": embarked_q,
            "Embarked_S": embarked_s,
        }

        df_entrada = pd.DataFrame([[entrada[f] for f in features]], columns=features)
        entrada_escalada = scaler.transform(df_entrada)

        probabilidad = float(
            modelo.predict(entrada_escalada, verbose=0).ravel()[0]
        )
        prediccion = int(probabilidad >= 0.5)

        st.divider()
        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Probabilidad estimada de supervivencia",
                f"{probabilidad * 100:.2f} %"
            )

        with r2:
            if prediccion == 1:
                st.success("Predicción: SOBREVIVE")
            else:
                st.warning("Predicción: NO SOBREVIVE")

        st.progress(probabilidad)

        with st.expander("Ver datos procesados"):
            st.dataframe(df_entrada, use_container_width=True)

        st.caption(
            "La salida representa una estimación del modelo entrenado sobre "
            "datos históricos del Titanic; no implica una relación causal."
        )

with tab_metricas:
    st.subheader("Desempeño de la red neuronal")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f'{metricas["accuracy"] * 100:.2f} %')
    m2.metric("Precision", f'{metricas["precision"] * 100:.2f} %')
    m3.metric("Recall", f'{metricas["recall"] * 100:.2f} %')
    m4.metric("F1-Score", f'{metricas["f1_score"] * 100:.2f} %')

    if "loss" in metricas:
        st.metric("Loss de prueba", f'{metricas["loss"]:.4f}')

    st.markdown(
        """
        **Interpretación breve**

        - **Accuracy:** proporción total de predicciones correctas.
        - **Precision:** confiabilidad de las predicciones positivas.
        - **Recall:** proporción de sobrevivientes reales identificados.
        - **F1-Score:** equilibrio entre Precision y Recall.
        """
    )

    matriz = np.array([[102, 8], [25, 44]])
    st.markdown("#### Matriz de confusión")
    st.dataframe(
        pd.DataFrame(
            matriz,
            index=["Real: No sobrevivió", "Real: Sobrevivió"],
            columns=["Pred: No sobrevivió", "Pred: Sobrevivió"]
        ),
        use_container_width=True
    )

with tab_comp:
    st.subheader("Comparación con los modelos del ACA 1")

    comparacion = pd.DataFrame(
        {
            "Modelo": [
                "Regresión Logística",
                "Árbol de Decisión",
                "Random Forest",
                "Red Neuronal",
            ],
            "Accuracy": [0.8045, 0.7765, 0.8101, metricas["accuracy"]],
            "Precision": [0.7833, 0.8085, 0.8571, metricas["precision"]],
            "Recall": [0.6812, 0.5507, 0.6087, metricas["recall"]],
            "F1-Score": [0.7287, 0.6552, 0.7119, metricas["f1_score"]],
        }
    )

    st.dataframe(
        comparacion.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }
        ),
        use_container_width=True
    )

    st.bar_chart(
        comparacion.set_index("Modelo")[
            ["Accuracy", "Precision", "Recall", "F1-Score"]
        ]
    )

    st.info(
        "La red neuronal obtuvo la mejor Accuracy del análisis, mientras "
        "Random Forest conservó la mayor Precision y Regresión Logística "
        "el mejor Recall y F1-Score."
    )

with tab_modelo:
    st.subheader("Arquitectura y configuración")

    arquitectura = metadata.get("arquitectura", {})
    capas = arquitectura.get("capas", [32, 16])
    dropout = arquitectura.get("dropout", [0.20, 0.10])
    lr = arquitectura.get("learning_rate", 0.001)

    st.markdown(
        f"""
        **Arquitectura seleccionada**

        `10 entradas → Dense {capas[0]} (ReLU) → Dropout {dropout[0]:.2f}
        → Dense {capas[1]} (ReLU) → Dropout {dropout[1]:.2f}
        → Dense 1 (Sigmoid)`

        **Learning rate:** `{lr}`

        La arquitectura se mantuvo compacta debido al tamaño del dataset.
        Durante el entrenamiento se utilizaron técnicas de regularización
        y parada temprana para controlar el sobreajuste.
        """
    )

    st.markdown("#### Variables de entrada")
    st.code("\\n".join(features))

    st.markdown("#### Resultado técnico")
    st.write(
        "La red neuronal mejora ligeramente la exactitud global frente a los "
        "modelos clásicos, pero no domina todas las métricas. Esto muestra "
        "que una arquitectura más avanzada no garantiza superioridad absoluta "
        "en un dataset pequeño y estructurado."
    )

st.divider()
st.caption(
    "ACA 2 · Especialización en Inteligencia Artificial · "
    "Corporación Unificada Nacional de Educación Superior (CUN)"
)
