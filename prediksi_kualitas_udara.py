from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# Fungsi untuk memuat model yang telah dilatih
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

# Fungsi untuk melakukan prediksi dengan model
def predict(model, input_data, scaler):
    # Ubah urutan input sesuai dengan kebutuhan model LSTM
    input_data = np.array(input_data).reshape(1, 3, 3)

    # Perform the prediction
    prediction = model.predict(input_data)

    # Ubah bentuk prediksi sesuai dengan kebutuhan Anda
    prediction = np.reshape(prediction, (prediction.shape[0]*3, 3))

    # Invers transformasi menggunakan scaler
    if scaler is not None and hasattr(scaler, 'inverse_transform'):
        prediction = scaler.inverse_transform(prediction)

    return prediction

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Memuat model
model = load_model()

# Tampilan Streamlit
st.title("Model Deployment with Streamlit")

# Slider untuk memasukkan nilai input
slider_value = st.slider("Enter AQI", min_value=0, max_value=400, step=1)

# Tombol untuk melakukan prediksi
if st.button("Predict"):
    # Menyusun input_data untuk prediksi
    input_data = [
        [slider_value/3, 0, 0],
        [0, 0, slider_value/3],
        [0, slider_value/3, 0]
    ]

    # Fitting scaler
    if not hasattr(scaler, 'n_samples_seen_'):
        scaler.fit(input_data)

    # Memanggil fungsi prediksi dengan input dari pengguna
    prediction = predict(model, input_data, scaler)

    # Menampilkan hasil prediksi dalam bentuk tabel
    prediction_df = pd.DataFrame(prediction, columns=['AQI', 'PM10', 'PM2.5'])
    st.table(prediction_df)