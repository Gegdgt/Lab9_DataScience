import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np

# Ruta donde se encuentran los archivos
data_path = 'C:/Users/manue/OneDrive/Escritorio/Data_Science/Lab9/'

# Cargar los datasets
precios_glp = pd.read_excel(data_path + 'Precios glp.xlsx')
diesel_data = pd.read_excel(data_path + 'diesel.xlsx')
regular_data = pd.read_excel(data_path + 'regular.xlsx')
super_data = pd.read_excel(data_path + 'super.xlsx')
importacion_data = pd.read_excel(data_path + 'importacion.xlsx')
consumo_data = pd.read_excel(data_path + 'consumo.xlsx')

# Limpiar y preparar los datos de GLP
precios_glp.columns = ["MES", "25 LBS.", "35 LBS.", "40 LBS.", "60 LBS.", "100 LBS."]
precios_glp["MES"] = pd.to_datetime(precios_glp["MES"], format="%b-%y", errors='coerce')
precios_glp.iloc[:, 1:] = precios_glp.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
precios_glp = precios_glp.dropna().reset_index(drop=True)

# Diccionario para traducir los meses al inglés
meses_traduccion = {
    "ENERO": "January", "FEBRERO": "February", "MARZO": "March", "ABRIL": "April", "MAYO": "May", 
    "JUNIO": "June", "JULIO": "July", "AGOSTO": "August", "SEPTIEMBRE": "September", "OCTUBRE": "October", 
    "NOVIEMBRE": "November", "DICIEMBRE": "December"
}

# Procesar los datos de Diesel, Regular, y Super
def procesar_datos(df):
    df = df[~df["MES/AÑO"].str.contains("Promedio", case=False, na=False)]
    df["MES/AÑO"] = df["MES/AÑO"].replace(meses_traduccion)
    df = df.melt(id_vars=["MES/AÑO"], var_name="Año", value_name="Precio")
    df["Año"] = df["Año"].astype(str)
    df["Fecha"] = pd.to_datetime(df["MES/AÑO"] + " " + df["Año"], format="%B %Y")
    return df[["Fecha", "Precio"]]

diesel_data_proc = procesar_datos(diesel_data)
regular_data_proc = procesar_datos(regular_data)
super_data_proc = procesar_datos(super_data)

# Procesar datos de importación
importacion_data['Fecha'] = pd.to_datetime(importacion_data['Fecha'], format='%Y-%m-%d')

# Procesar datos de consumo
consumo_data['Fecha'] = pd.to_datetime(consumo_data['Fecha'], format='%Y-%m-%d')

# ==================
# Diseño de la interfaz con barra lateral
st.sidebar.title("Opciones de Interacción")

# Selección de tipo de combustible en la barra lateral
selected_fuel = st.sidebar.radio("Selecciona el tipo de combustible para resaltar", ["Diesel", "Regular", "Super"])

# Selección de modelo de predicción
selected_model = st.sidebar.selectbox("Selecciona el modelo para predicciones", ["ARIMA", "Exponential Smoothing", "Simple Moving Average"])

# ==================
# Colores
primary_color = "#006699"  # Azul
secondary_color = "#339966"  # Verde
highlight_color = "#99CC33"  # Color más claro para destacar
background_color = "white"

# ==================
# Visualización interactiva: Evolución histórica de precios
st.subheader("Evolución Histórica del Precio de Combustibles")

fuel_mapping = {
    "Diesel": diesel_data_proc,
    "Regular": regular_data_proc,
    "Super": super_data_proc
}

highlight_fuel = fuel_mapping[selected_fuel]

fig_fuel_hist = go.Figure()

for fuel, data in zip(["Diesel", "Regular", "Super"], [diesel_data_proc, regular_data_proc, super_data_proc]):
    color = primary_color if fuel == selected_fuel else secondary_color
    opacity_value = 1.0 if fuel == selected_fuel else 0.2
    fig_fuel_hist.add_trace(go.Scatter(x=data["Fecha"], y=data["Precio"],
                                       mode='lines', name=fuel, opacity=opacity_value,
                                       line=dict(width=3 if fuel == selected_fuel else 1, color=color)))

fig_fuel_hist.update_layout(
    title="Evolución Histórica del Precio de Combustibles",
    xaxis_title="Fecha",
    yaxis_title="Precio (Q)",
    plot_bgcolor=background_color,
    font=dict(family="Arial", size=12, color="black"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True)
)

st.plotly_chart(fig_fuel_hist)

# ==================
# Visualización interactiva: Consumo total por mes con selección de combustible
st.subheader("Consumo Total por Mes")

# Filtrar el consumo basado en la selección del usuario
if selected_fuel == "Diesel":
    fuel_col = "Diesel alto azufre"  # Cambiar a la columna correspondiente para diesel
elif selected_fuel == "Regular":
    fuel_col = "Gasolina regular"  # Cambiar a la columna correcta para gasolina regular
else:
    fuel_col = "Gasolina superior"  # Cambiar a la columna correcta para gasolina superior

# Visualización interactiva del consumo por mes
fig_consumo = px.line(consumo_data, x='Fecha', y=fuel_col, title=f"Consumo Total de {selected_fuel} por Mes",
                      color_discrete_sequence=[highlight_color])
fig_consumo.update_layout(plot_bgcolor=background_color, xaxis_title="Fecha", yaxis_title="Consumo (Litros)")
st.plotly_chart(fig_consumo)

# ==================
# Visualización interactiva: Diferencias porcentuales de precios de combustibles
st.subheader("Diferencias porcentuales entre precios de combustibles")

def calcular_diferencias(df):
    df["Diff"] = df["Precio"].pct_change().dropna()
    return df

diesel_diff = calcular_diferencias(diesel_data_proc)
regular_diff = calcular_diferencias(regular_data_proc)
super_diff = calcular_diferencias(super_data_proc)

fig_diff_fuel = go.Figure()

for fuel, data in zip(["Diesel", "Regular", "Super"], [diesel_diff, regular_diff, super_diff]):
    color = primary_color if fuel == selected_fuel else secondary_color
    opacity_value = 1.0 if fuel == selected_fuel else 0.2
    fig_diff_fuel.add_trace(go.Scatter(x=data["Fecha"], y=data["Diff"],
                                       mode='lines', name=f'{fuel} - % Cambio', opacity=opacity_value,
                                       line=dict(width=3 if fuel == selected_fuel else 1, color=color)))

fig_diff_fuel.update_layout(
    title="Diferencias porcentuales entre precios de combustibles",
    xaxis_title="Fecha",
    yaxis_title="Diferencia Porcentual (%)",
    plot_bgcolor=background_color,
    font=dict(family="Arial", size=12, color="black"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True)
)

st.plotly_chart(fig_diff_fuel)

# ==================
# Gráfico interactivo: Importaciones Totales por Producto y Mes
st.subheader("Importaciones Totales por Producto y Mes")

# Selección del producto a visualizar
productos_disponibles = importacion_data.columns.tolist()[1:-1]  # Omitir la columna 'Fecha' y 'Total importación'
selected_product = st.sidebar.selectbox("Selecciona el producto a visualizar", productos_disponibles)

# Gráfico interactivo de barras
fig_importacion_producto = px.bar(importacion_data, x='Fecha', y=selected_product, title=f"Importaciones de {selected_product} por Mes",
                                  labels={selected_product: "Importación (Toneladas)"},
                                  color_discrete_sequence=[primary_color])
fig_importacion_producto.update_layout(plot_bgcolor=background_color, xaxis_title="Fecha", yaxis_title="Importación (Toneladas)")
st.plotly_chart(fig_importacion_producto)

# ==================
# Predicciones con múltiples modelos
st.subheader(f"Predicciones utilizando el modelo seleccionado: {selected_model}")

precios_glp_ts = precios_glp.set_index("MES")["25 LBS."]

# Predicción con ARIMA
if selected_model == "ARIMA":
    modelo_arima_glp = ARIMA(precios_glp_ts, order=(5, 1, 0))
    modelo_arima_glp_fit = modelo_arima_glp.fit()
    pred_arima_glp = modelo_arima_glp_fit.forecast(steps=12)
    pred_dates = pd.date_range(start=precios_glp_ts.index[-1], periods=12, freq='M')
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=precios_glp_ts.index, y=precios_glp_ts, mode='lines', name='Datos Observados', line=dict(color=primary_color)))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_arima_glp, mode='lines', name='Predicción ARIMA', line=dict(dash='dash', color=highlight_color)))
elif selected_model == "Exponential Smoothing":
    es_model = ExponentialSmoothing(precios_glp_ts, trend='add', seasonal='add', seasonal_periods=12)
    es_fit = es_model.fit()
    pred_es = es_fit.forecast(12)
    pred_dates = pd.date_range(start=precios_glp_ts.index[-1], periods=12, freq='M')
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=precios_glp_ts.index, y=precios_glp_ts, mode='lines', name='Datos Observados', line=dict(color=primary_color)))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_es, mode='lines', name='Predicción Exponential Smoothing', line=dict(dash='dash', color=highlight_color)))

# Predicción con Simple Moving Average
elif selected_model == "Simple Moving Average":
    sma_window = 12
    precios_glp_sma = precios_glp_ts.rolling(window=sma_window).mean().dropna()
    pred_sma = np.repeat(precios_glp_sma.iloc[-1], 12)
    pred_dates = pd.date_range(start=precios_glp_ts.index[-1], periods=12, freq='M')
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=precios_glp_ts.index, y=precios_glp_ts, mode='lines', name='Datos Observados', line=dict(color=primary_color)))
    fig_pred.add_trace(go.Scatter(x=precios_glp_sma.index, y=precios_glp_sma, mode='lines', name='Simple Moving Average', line=dict(dash='dash', color=highlight_color)))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_sma, mode='lines', name='Predicción SMA', line=dict(dash='dot', color=secondary_color)))

# Configuración final de la gráfica
fig_pred.update_layout(
    title=f"Predicción de Precios con {selected_model}",
    xaxis_title="Fecha",
    yaxis_title="Precio (Q)",
    plot_bgcolor=background_color,
    font=dict(family="Arial", size=12, color="black"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True)
)

st.plotly_chart(fig_pred)