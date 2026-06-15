import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# HALAMAN UTAMA
st.set_page_config(
    page_title="Anime AI Detector", 
    layout="wide" 
)

# Kustomisasi CSS
st.markdown("""
    <style>
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# 2. SIDEBAR INFORMASI
with st.sidebar:
    st.title("Tentang Sistem")
    st.info(
        "Implementasi algoritma **Convolutional Neural Network (CNN)** "
        "menggunakan arsitektur **MobileNetV3 Large**."
    )
    st.write("---")
    st.write("**Cara Penggunaan:**")
    st.write("1. Unggah gambar ilustrasi anime format JPG/PNG/JPEG.")
    st.write("2. Klik tombol **Mulai Analisis**.")
    st.write("3. Sistem akan memprediksi keaslian gambar berdasarkan ekstraksi fitur visual.")
    st.write("---")
    st.caption("Penulisan Ilmiah - Universitas Gunadarma")

# 3. HEADER DAN JUDUL
st.title("Anime Illustration AI Detector")
st.markdown("Deteksi keaslian karya ilustrasi digital bergaya anime. Cari tahu apakah gambar di bawah ini murni goresan tangan manusia atau hasil *Generative AI*.")
st.write("---")

# 4. PEMUATAN MODEL (CACHING)
@st.cache_resource
def load_model():
    model_path = 'mobilenetv3_large_anime_ai_v1.keras' 
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file '{model_path}' berada di folder yang sama. Detail error: {e}")

# 5. LOGIKA APLIKASI DAN ANTARMUKA
# Area Upload Gambar
uploaded_file = st.file_uploader("Pilih ilustrasi anime untuk dianalisis...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Membaca gambar dengan OpenCV (menghindari penggunaan PIL)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Membagi layar menjadi 2 kolom (Rasio 1:1)
    col1, col2 = st.columns([1, 1])
    
    # KOLOM KIRI: Menampilkan Gambar
    with col1:
        st.markdown("### Gambar Masukan")
        st.image(img_rgb, caption='Citra yang diunggah siap diproses', use_container_width=True)
        
    # KOLOM KANAN: Hasil Analisis
    with col2:
        st.markdown("### Hasil Analisis")
        
        if st.button('Mulai Analisis'):
            with st.spinner('Mengekstraksi karakteristik fitur visual...'):
                # Tahap Preprocessing
                img_resized = cv2.resize(img_rgb, (224, 224))
                img_array = img_resized / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Tahap Prediksi
                prediction = model.predict(img_array)[0][0]
                
                st.markdown("#### Kesimpulan Prediksi:")
                
                # Logika output klasifikasi biner
                if prediction > 0.5:
                    hasil = "BUATAN MANUSIA (Non-AI)"
                    confidence = prediction * 100
                    st.success(f"**{hasil}**")
                else:
                    hasil = "HASIL GENERATIF AI"
                    confidence = (1 - prediction) * 100
                    st.error(f"**{hasil}**")
                    
                # Menampilkan Confidence Score dengan Progress Bar
                st.markdown("#### Tingkat Keyakinan (*Confidence Score*):")
                progress_bar = st.progress(0)
                progress_bar.progress(int(confidence))
                st.write(f"Model memiliki tingkat keyakinan sebesar **{confidence:.2f}%** terhadap hasil tersebut.")
                
                # Detail teknis yang bisa di-expand
                with st.expander("Lihat Detail Komputasi"):
                    st.write(f"- **Resolusi Pemrosesan:** 224 x 224 Piksel")
                    st.write(f"- **Nilai Probabilitas Mentah (Sigmoid):** {prediction:.6f}")
                    st.write(f"- **Fungsi Aktivasi Output:** Sigmoid")