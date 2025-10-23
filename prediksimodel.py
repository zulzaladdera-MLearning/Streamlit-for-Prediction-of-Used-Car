import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide")
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #1e1e1e;
            color: white;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
        }
        [data-testid="stImage"] img {
            border-radius: 50%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Sidebar: Profil
# ===============================
with st.sidebar:
    st.image("7228173335788045339_avatar.png.jpg", width=180)
    st.markdown("## Zulza Laddera Aripin")
    st.markdown("👨‍💻 *Data Science Enthusiast*")
    st.markdown("---")

# ====================================================
# 🧩 Konfigurasi Awal
# ====================================================
st.set_page_config(page_title="Used Car Price Prediction", layout="centered")
st.title("🚗 Prediksi Harga Mobil Second")
st.write("Memperkirakan harga mobil bekas berdasarkan spesifikasi.")

# ====================================================
# 🧠 Muat Model & Encoder
# ====================================================
try:
    model = joblib.load("bmw_price_model2.pkl")
    encoders = joblib.load("encoders2.pkl")
    # st.success("✅ Model dan encoder berhasil dimuat.")
except Exception as e:
    st.error("❌ Model belum ditemukan. Jalankan script training XGBoost terlebih dahulu.")
    st.stop()

# ====================================================
# 📥 Input Pengguna
# ====================================================
st.header("Masukkan Spesifikasi Mobil")

col1, col2 = st.columns(2)

with col1:
    model_input = st.selectbox("Model Mobil", sorted(encoders['model'].classes_))
    year_input = st.number_input("Tahun Produksi", min_value=1990, max_value=2025, value=2020)
    engine_input = st.number_input("Ukuran Mesin (L)", min_value=0.5, max_value=6.0, value=2.0)

with col2:
    mileage_input = st.number_input("Jarak Tempuh (miles)", min_value=0, value=20000)
    tax_input = st.number_input("Pajak (Pound Sterling)", min_value=0, value=150)
    mpg_input = st.number_input("Efisiensi (MPG)", min_value=0.0, value=40.0)

fuel_input = st.selectbox("Tipe Bahan Bakar", sorted(encoders['fuelType'].classes_))
ownership_input = st.selectbox(
    "Jumlah Kepemilikan (Ownership)",
    options=[1, 2, 3],
    index=0,
    help="Kepemilikan keberapa mobil ini (1 = tangan pertama, dst.)"
)

# ====================================================
# 🔮 Prediksi
# ====================================================
if st.button("Prediksi Harga"):
    try:
        # Encode input kategorikal
        encoded_model = encoders['model'].transform([model_input])[0]
        encoded_fuel = encoders['fuelType'].transform([fuel_input])[0]

        # Konversi tax ke rupiah
        exchange_rate = 22208.42
        tax_rupiah = tax_input * exchange_rate

        # Susun input sesuai urutan fitur pada training
        input_data = pd.DataFrame([{
            'year': year_input,
            'mileage': mileage_input,
            'tax': tax_rupiah,  # sudah dikonversi
            'mpg': mpg_input,
            'engineSize': engine_input,
            'fuelType': encoded_fuel,
            'model': encoded_model,
            'ownership': ownership_input
        }])

        # Prediksi langsung (tanpa scaling)
        predicted_price = model.predict(input_data)[0]

        st.success(f"💰 Prediksi Harga Mobil BMW: **Rp {predicted_price:,.0f}**")
        st.caption("Perkiraan harga berdasarkan model XGBoost Regressor tanpa normalisasi fitur.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")