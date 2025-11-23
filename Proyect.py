# Eduardo Nolasco Gòmez
# Data Mining - Proyecto Final

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Cargamos el dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# Preprocesamooooss
X = df[iris.feature_names]
y = df['species']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Modelo
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


# STREAMLIT UI
st.set_page_config(page_title="Iris Species Classification", layout="wide")
st.title("Iris Species Classification Dashboard")
st.sidebar.markdown("**Autores:** Tu Nombre Completo  \nCompañero: Nombre Completo")
st.write("Modelo entrenado para clasificar especies de Iris usando 4 características.")

# Métricas
st.header("Métricas del Modelo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1-Score", f"{f1:.3f}")

# Predicción interactiva
st.header("Hacer una Predicción")
with st.form("predict_form"):
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    submit = st.form_submit_button("Predecir Especie")

if submit:
    new_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(new_data)[0]
    st.success(f"La especie predicha es: **{prediction}**")

    # 3D scatter (usamos las 3 primeras features para 3D)
    fig_3d = px.scatter_3d(
        df,
        x=iris.feature_names[0],
        y=iris.feature_names[1],
        z=iris.feature_names[2],
        color="species",
        opacity=0.7,
        title="Posición de la nueva muestra en el espacio 3D"
    )

    fig_3d.add_scatter3d(
        x=[sepal_length],
        y=[sepal_width],
        z=[petal_length],
        mode="markers",
        marker=dict(size=8, symbol="diamond", line=dict(width=1)),
        name="Nueva muestra"
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# Visualizaciones adicionales
st.header("Distribuciones de las características")
fig_hist = px.histogram(df.melt(id_vars="species", value_vars=iris.feature_names),
                        x="value", color="species", facet_col="variable",
                        title="Distribución de features por especie",
                        labels={"value":"cm"})
st.plotly_chart(fig_hist, use_container_width=True)

st.write("Repositorio: (añadir link en README).")
