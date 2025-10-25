import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt

st.set_page_config(page_title="Regresión: calificación vs horas", page_icon="📘", layout="centered")

# -------------------- DEMO DATA MÁS REALISTA --------------------
def make_demo_df(n=80, seed=42):
    """
    Simula datos de estudiantes:
    - horas: 0 a 10 horas de estudio semanales
    - calificación: 35 a 100, con variabilidad realista
    """
    rng = np.random.default_rng(seed)
    horas = rng.uniform(0, 10, n)
    calif = 35 + 6.2 * horas + rng.normal(0, 5, n)  # pendiente realista
    calif = np.clip(calif, 30, 100)
    return pd.DataFrame({"horas": np.round(horas, 2), "calificacion": np.round(calif, 2)})

if "df" not in st.session_state:
    st.session_state.df = make_demo_df()
if "source" not in st.session_state:
    st.session_state.source = "DEMO"

st.title("📘 Regresión lineal: calificación vs horas de estudio Hecho Por Patricio C")
st.caption("Sube un CSV con columnas **horas** y **calificacion** para usar tus propios datos.")
st.markdown(f"**Fuente actual:** `{st.session_state.source}`")

# -------------------- CARGA CSV O RESTAURAR --------------------
c1, c2 = st.columns([3,1])
with c1:
    up = st.file_uploader("CSV (columnas: horas, calificacion)", type=["csv"])
with c2:
    if st.button("Restaurar DEMO"):
        st.session_state.df = make_demo_df()
        st.session_state.source = "DEMO"
        st.success("Datos restaurados.")

if up is not None:
    try:
        df_new = pd.read_csv(up)
        req = ["horas", "calificacion"]
        missing = [c for c in req if c not in df_new.columns]
        if missing:
            st.error(f"Faltan columnas: {missing}")
            st.stop()
        df_new = df_new[req].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df_new) < 10:
            st.error("Se necesitan al menos 10 filas válidas.")
            st.stop()
        st.session_state.df = df_new.reset_index(drop=True)
        st.session_state.source = "CSV"
        st.success("CSV cargado correctamente.")
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        st.stop()

df = st.session_state.df

# -------------------- PREVIA --------------------
st.subheader("Vista previa")
st.dataframe(df.head(15), use_container_width=True)

# -------------------- ENTRENAMIENTO SENCILLO --------------------
st.subheader("Entrenamiento del modelo")
test_size = st.slider("Proporción de test", 0.1, 0.5, 0.2, 0.05)

X = df[["horas"]].to_numpy(dtype=float)
y = df["calificacion"].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

b1 = float(model.coef_[0])
b0 = float(model.intercept_)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
r2 = float(r2_score(y_test, y_pred))

st.subheader("Ecuación del modelo")
st.latex(r"\text{calificación} = \beta_0 + \beta_1 \cdot \text{horas}")
st.write(f"**β₀:** {b0:,.2f}  |  **β₁:** {b1:,.2f}  (puntos por hora de estudio)")

c3, c4 = st.columns(2)
c3.metric("R² (test)", f"{r2:.4f}")
c4.metric("RMSE (test)", f"{rmse:.2f}")

# -------------------- GRÁFICA --------------------
grid = pd.DataFrame({"horas": np.linspace(X.min(), X.max(), 100).ravel()})
grid["calif_pred"] = model.predict(grid[["horas"]])

pts = alt.Chart(df).mark_circle(size=60, opacity=0.6, color="#4B9CD3").encode(
    x=alt.X("horas", title="Horas de estudio"),
    y=alt.Y("calificacion", title="Calificación (0–100)"),
    tooltip=["horas", "calificacion"]
)
linea = alt.Chart(grid).mark_line(color="red").encode(
    x="horas", y=alt.Y("calif_pred", title="Calificación (predicha)")
)
st.subheader("Datos y recta ajustada")
st.altair_chart(pts + linea, use_container_width=True)

# -------------------- PREDICCIÓN --------------------
st.header("Predicción rápida")
h = st.number_input("Horas de estudio (una sola)", value=float(np.median(X)))
pred = model.predict(np.array([[h]])).item()
pred = float(np.clip(pred, 0, 100))
st.success(f"Predicción de calificación para {h:,.2f} horas: **{pred:,.2f} puntos**")

# -------------------- FIRMA --------------------
st.markdown("---")
st.caption("Signature: Patricio C")
