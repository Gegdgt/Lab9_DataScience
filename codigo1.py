import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ruta donde se encuentran los archivos
data_path = 'C:/Users/gegdg/OneDrive/Documentos/.UVG/Anio4/Ciclo2/Data_Science/Lab9/'

# Cargar los datasets
diesel_data = pd.read_excel(data_path + 'diesel.xlsx')
regular_data = pd.read_excel(data_path + 'regular.xlsx')
super_data = pd.read_excel(data_path + 'super.xlsx')
importacion_data = pd.read_excel(data_path + 'importacion.xlsx')
consumo_data = pd.read_excel(data_path + 'consumo.xlsx')

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
# Barra lateral de selección de combustible
st.sidebar.title("Opciones de Interacción")
selected_fuel = st.sidebar.radio("Selecciona el tipo de combustible", ["Diesel", "Regular", "Super"])

# ==================
# Colores
primary_color = "#006699"  # Azul
secondary_color = "#339966"  # Verde
highlight_color = "#99CC33"  # Color más claro para destacar
background_color = "white"

# ==================
# Visualización interactiva: Consumo total por mes
st.subheader("Consumo Total por Mes")

# Filtrar el consumo basado en la selección del usuario
if selected_fuel == "Diesel":
    fuel_col = "Diesel alto azufre"
elif selected_fuel == "Regular":
    fuel_col = "Gasolina regular"
else:
    fuel_col = "Gasolina superior"

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
# Visualización de Importaciones con gráfica interactiva
st.subheader("Gráfico interactivo: Importaciones Totales por Producto")

importacion_totales = importacion_data.groupby('Fecha')['Total importación'].sum().reset_index()
fig_importacion_totales = px.bar(importacion_totales, x='Fecha', y='Total importación', title="Importaciones Totales por Mes",
                                 color_discrete_sequence=[primary_color])
fig_importacion_totales.update_layout(plot_bgcolor=background_color)
st.plotly_chart(fig_importacion_totales)
