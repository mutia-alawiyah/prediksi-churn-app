import streamlit as st
import joblib
import numpy as np

# 1. Judul Aplikasi
st.title("Aplikasi Prediksi Churn Pelanggan 🎧")
st.write("Masukkan data pelanggan di bawah ini untuk memprediksi apakah pelanggan akan Churn atau Tidak.")

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('churn_spotify.pkl')

model = load_model()

# 3. Form Input Data di Sidebar
st.sidebar.header("Input Data Pelanggan")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age (Usia)", min_value=10, max_value=100, value=25)
country = st.sidebar.selectbox("Country", ["USA", "Canada", "UK", "Germany", "France"])
subscription_type = st.sidebar.selectbox("Subscription Type", ["Free", "Basic", "Premium"])
listening_time = st.sidebar.number_input("Listening Time (menit/hari)", min_value=0, max_value=1440, value=60)
songs_played_per_day = st.sidebar.number_input("Songs Played per Day", min_value=0, max_value=500, value=20)
skip_rate = st.sidebar.slider("Skip Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
device_type = st.sidebar.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
ads_listened_per_week = st.sidebar.number_input("Ads Listened per Week", min_value=0, max_value=100, value=5)
offline_listening = st.sidebar.selectbox("Offline Listening", [0, 1], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)")

# 4. Tombol Prediksi
if st.button("Prediksi Status Churn"):
    # Encoding sederhana
    gender_enc = 1 if gender == "Male" else 0
    
    # Kumpulkan semua input (pastikan semua variabel terdefinisi di atas)
    features = np.array([[
        gender_enc, 
        age, 
        listening_time, 
        songs_played_per_day, 
        skip_rate, 
        ads_listened_per_week, 
        offline_listening
    ]])
    
    # Lakukan prediksi
    prediction = model.predict(features)[0]
    
    # Target label (0: Tidak Churn, 1: Churn)
    target_names = ['Tidak Churn (Tetap Berlangganan)', 'Churn (Berhenti Berlangganan)']
    hasil_prediksi = target_names[prediction]
    
    if prediction == 1:
        st.error(f"Hasil Prediksi: **{hasil_prediksi}**")
    else:
        st.success(f"Hasil Prediksi: **{hasil_prediksi}**")
