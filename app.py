import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import altair as alt
from io import StringIO

st.set_page_config(page_title="Regresión: renta vs m²", page_icon="🏠", layout="centered")

# -------------------- DEMO DATA --------------------
def make_demo_df(n=180, beta0=2500.0, beta1=150.0, noise=2500.0, seed=7):
    """
    Crea un dataset sintético de rentas en MXN (renta) y metros cuadrados (m2).
    renta = beta0 + beta1 * m2 + ruido
    """
    rng = np.random.default_rng(seed)
    m2 = np.linspace(20, 200, n)
    renta = beta0 + beta1 * m2 + rng.normal(0, noise, size=n)
    renta = np.clip(renta, 1500, None)  # sin rentas negativas/raras
    return pd.DataFrame({"m2": m2, "renta": renta})

if "train_df" not in st.session_state:
    st.session_state.train_df = make_demo_df()
if "source" not in st.session_state:
    st.session_state.source = "DEMO"

# -------------------- UI HEADER --------------------
st.title("🏠 Regresión lineal simple: renta (MXN) en función de m²")
st.caption("Sube un CSV con columnas **m2** y **renta** para reemplazar los datos. Métricas, gráfica y predicción se actualizan al vuelo.")
st.markdown(f"**Fuente de datos actual:** `{st.session_state.source}`")

# -------------------- CONTROLES DE ORIGEN --------------------
colA, colB, colC = st.columns([3, 1, 1])
with colA:
    up = st.file_uploader("Reemplazar datos con CSV (columnas: m2, renta)", type=["csv"])
with colB:
    if st.button("Restaurar DEMO"):
        st.session_state.train_df = make_demo_df()
        st.session_state.source = "DEMO"
        st.success("Datos restaurados a DEMO.")
with colC:
    # Descargar plantilla CSV
    tpl = pd.DataFrame({"m2": [45, 60, 80, 120], "renta": [7000, 9500, 12000, 18000]})
    csv_buf = StringIO()
    tpl.to_csv(csv_buf, index=False)
    st.download_button("Descargar plantilla", csv_buf.getvalue(), file_name="plantilla_renta_m2.csv", mime="text/csv")

# Cargar CSV del usuario
if up is not None:
    try:
        df_new = pd.read_csv(up)
        missing = [c for c in ["m2", "renta"] if c not in df_new.columns]
        if missing:
            st.error(f"Faltan columnas requeridas: {missing}. El CSV debe tener 'm2' y 'renta'.")
            st.stop()
        df_new = df_new[["m2", "renta"]].copy()
        df_new["m2"] = pd.to_numeric(df_new["m2"], errors="coerce")
        df_new["renta"] = pd.to_numeric(df_new["renta"], errors="coerce")
        before = len(df_new)
        df_new = df_new.dropna()
        if len(df_new) < 10:
            st.error("Muy pocos datos numéricos tras limpiar. Revisa tu CSV (mínimo 10 filas válidas).")
            st.stop()
        if len(df_new) < before:
            st.info(f"Se descartaron {before - len(df_new)} filas no numéricas o vacías.")
        st.session_state.train_df = df_new.reset_index(drop=True)
        st.session_state.source = "CSV"
        st.success("Datos reemplazados por tu CSV. La app ya usa tu tabla para todo.")
    except Exception as e:
        st.error(f"Error al leer CSV: {e}")
        st.stop()

df = st.session_state.train_df

# -------------------- PREVIA --------------------
st.subheader("Vista previa de los datos actuales")
st.dataframe(df.head(20), use_container_width=True)

# -------------------- ENTRENAMIENTO --------------------
st.subheader("Entrenamiento del modelo")
test_size = st.slider("Proporción de test", 0.1, 0.5, 0.25, 0.05)
use_norm = st.checkbox(
    "Normalizar X e Y (z-score)",
    value=False,
    help="El modelo se entrena en escala estandarizada, pero se reporta en escala original."
)

X_raw = df[["m2"]].to_numpy(dtype=float)
Y_raw = df["renta"].to_numpy(dtype=float)

X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
    X_raw, Y_raw, test_size=test_size, random_state=42
)

if use_norm:
    x_mu, x_sd = X_train_raw.mean(), X_train_raw.std(ddof=0)
    y_mu, y_sd = Y_train_raw.mean(), Y_train_raw.std(ddof=0)
    x_sd = x_sd if x_sd > 0 else 1.0
    y_sd = y_sd if y_sd > 0 else 1.0

    X_train = (X_train_raw - x_mu) / x_sd
    Y_train = (Y_train_raw - y_mu) / y_sd
    X_test  = (X_test_raw  - x_mu) / x_sd

    model = LinearRegression().fit(X_train, Y_train)
    y_pred_norm = model.predict(X_test)
    y_pred = y_pred_norm * y_sd + y_mu

    a1 = float(model.coef_[0])
    a0 = float(model.intercept_)
    b1 = float((y_sd * a1) / x_sd)
    b0 = float(y_mu + y_sd * a0 - b1 * x_mu)
else:
    model = LinearRegression().fit(X_train_raw, Y_train_raw)
    y_pred = model.predict(X_test_raw)
    b1 = float(model.coef_[0])
    b0 = float(model.intercept_)

# -------------------- MÉTRICAS --------------------
mse = mean_squared_error(Y_test_raw, y_pred)
rmse = float(np.sqrt(mse))
r2 = float(r2_score(Y_test_raw, y_pred))

st.subheader("Ecuación del modelo (escala original)")
st.latex(r"\text{renta (MXN)} = \beta_0 + \beta_1 \cdot \text{m}^2")
st.write(f"**β₀:** {b0:,.4f} | **β₁:** {b1:,.4f}  (MXN por m²)")

m1, m2 = st.columns(2)
m1.metric("R² (test)", f"{r2:.4f}")
m2.metric("RMSE (test)", f"${rmse:,.2f}")

# -------------------- GRÁFICA --------------------
grid = pd.DataFrame({"m2": np.linspace(X_raw.min(), X_raw.max(), 100)})
if use_norm:
    denom = X_train_raw.std(ddof=0) if X_train_raw.std(ddof=0) > 0 else 1.0
    z = (grid[["m2"]].to_numpy(dtype=float) - X_train_raw.mean()) / denom
    pred = model.predict(z)
    grid["renta"] = pred * (Y_train_raw.std(ddof=0) if Y_train_raw.std(ddof=0) > 0 else 1.0) + Y_train_raw.mean()
else:
    grid["renta"] = model.predict(grid[["m2"]])

scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X("m2", title="m²"),
    y=alt.Y("renta", title="Renta (MXN)"),
    tooltip=[alt.Tooltip("m2", title="m²"), alt.Tooltip("renta", title="Renta (MXN)")]
)
line = alt.Chart(grid).mark_line().encode(x="m2", y="renta")

st.subheader("Ajuste del modelo con los datos actuales")
st.altair_chart(scatter + line, use_container_width=True)

# -------------------- PREDICCIÓN --------------------
st.header("Predicción con dato nuevo")
x_new = st.number_input("m² (valor único)", value=float(np.median(X_raw)))
if use_norm:
    denom = X_train_raw.std(ddof=0) if X_train_raw.std(ddof=0) > 0 else 1.0
    xz = (np.array([[x_new]]) - X_train_raw.mean()) / denom
    y_new = model.predict(xz).item()
    y_new = y_new * (Y_train_raw.std(ddof=0) if Y_train_raw.std(ddof=0) > 0 else 1.0) + Y_train_raw.mean()
else:
    y_new = model.predict(np.array([[x_new]])).item()

st.success(f"Predicción de renta para {x_new:,.2f} m²: **${y_new:,.2f}**")

# -------------------- FOOTER / FIRMA --------------------
st.markdown("---")
st.caption("Hecho con ❤️ en Streamlit — **Firma: Patricio C**")
