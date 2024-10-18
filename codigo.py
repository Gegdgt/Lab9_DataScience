import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Cargar los datasets
data_path = 'C:/Users/manue/OneDrive/Escritorio/Data_Science/Lab9/'
precios_glp = pd.read_excel(data_path + 'Precios glp.xlsx')

# Limpiar y preparar los datos
precios_glp.columns = ["MES", "25 LBS.", "35 LBS.", "40 LBS.", "60 LBS.", "100 LBS."]
precios_glp["MES"] = pd.to_datetime(precios_glp["MES"], format="%b-%y", errors='coerce')
precios_glp.iloc[:, 1:] = precios_glp.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
precios_glp = precios_glp.dropna().reset_index(drop=True)

# Mostrar las primeras filas del dataset
st.write("### Datos de Precios GLP:")
st.write(precios_glp.head())

# Visualización 1: Evolución histórica del precio del GLP (25 LBS) - Dinámica con Plotly
st.subheader("Evolución Histórica del Precio del GLP (25 LBS)")
fig = px.line(precios_glp, x='MES', y='25 LBS.', title='Evolución del Precio GLP (25 LBS)')
fig.update_layout(xaxis_title='Fecha', yaxis_title='Precio (Q)')
st.plotly_chart(fig)

# Predicción LSTM
st.subheader("Predicción LSTM para Precios GLP")
precios_glp_ts = precios_glp.set_index("MES")["25 LBS."]

# Escalar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
precios_glp_scaled = scaler.fit_transform(precios_glp_ts.values.reshape(-1, 1))
train_size = int(len(precios_glp_scaled) * 0.8)
train_data = precios_glp_scaled[:train_size]
test_data = precios_glp_scaled[train_size:]

# Verificar si hay suficientes datos para la predicción
if len(test_data) < 10:
    st.error("No hay suficientes datos para la predicción con LSTM.")
else:
    # Preparar datos para LSTM
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Redimensionar datos
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Definir el modelo LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compilar y entrenar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10)

    # Predecir con el LSTM
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Visualizar la predicción LSTM con Plotly (Dinámica)
    lstm_dates = precios_glp_ts.index[-len(predictions):]
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=precios_glp_ts.index[-len(test_data):], y=scaler.inverse_transform(test_data), mode='lines', name='Datos Observados'))
    fig_lstm.add_trace(go.Scatter(x=lstm_dates, y=predictions.flatten(), mode='lines', name='Predicción LSTM', line=dict(dash='dash')))
    fig_lstm.update_layout(title='Predicción LSTM para Precios GLP', xaxis_title='Fecha', yaxis_title='Precio (Q)')
    st.plotly_chart(fig_lstm)
