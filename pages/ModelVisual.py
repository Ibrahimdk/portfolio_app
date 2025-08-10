import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Konfigurasi Halaman
st.set_page_config(page_title="Model Visualization", layout="wide")
st.title("📊 Visualisasi Data dan Performa Model")

# Fungsi untuk memuat data dengan caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("File 'data_cleaned.csv' tidak ditemukan. Pastikan file tersebut berada di folder yang sama dengan aplikasi Streamlit Anda.")
        return None

df = load_data()

if df is not None:
    # Opsi Pilihan di Sidebar
    st.sidebar.title("Menu Visualisasi")
    viz_choice = st.sidebar.selectbox(
        "Pilih visualisasi yang ingin ditampilkan:",
        ("Distribusi Label Dataset", "Analisis Teks (Word Cloud & Frekuensi)", "Performa Model (Learning Curves & Confusion Matrix)")
    )
    st.sidebar.info("Visualisasi ini didasarkan pada data dan model yang telah dilatih di notebook 'Modelling_FraudGuard.ipynb'.")

    # --- Tampilan Utama berdasarkan Pilihan ---

    if viz_choice == "Distribusi Label Dataset":
        st.header("Distribusi Label pada Dataset")
        st.write("Visualisasi ini menunjukkan jumlah pesan untuk setiap kategori (normal, penipuan, promo) setelah proses augmentasi data untuk menyeimbangkan kelas.")

        label_counts = df['label'].value_counts().reset_index()
        label_counts.columns = ['Label', 'Jumlah']

        fig = px.bar(label_counts,
                     x='Label',
                     y='Jumlah',
                     title="Jumlah Pesan per Kategori",
                     color='Label',
                     text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Contoh Data")
        st.dataframe(df.sample(5))

    elif viz_choice == "Analisis Teks (Word Cloud & Frekuensi)":
        st.header("Analisis Teks dari Dataset")
        st.write("Melihat kata-kata yang paling sering muncul dalam keseluruhan dataset setelah dibersihkan.")

        # --- Word Cloud ---
        st.subheader("Word Cloud")
        all_words = ' '.join(df['Clean_Teks'].dropna())
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # --- Frekuensi Kata Teratas ---
        st.subheader("Top 10 Frekuensi Kata")
        word_freq = Counter(all_words.split())
        common_words_df = pd.DataFrame(word_freq.most_common(10), columns=['Kata', 'Frekuensi'])

        fig_freq = px.bar(common_words_df,
                          x='Frekuensi',
                          y='Kata',
                          orientation='h',
                          title="Top 10 Kata yang Paling Sering Muncul",
                          text_auto=True)
        fig_freq.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_freq, use_container_width=True)


    elif viz_choice == "Performa Model (Learning Curves & Confusion Matrix)":
        st.header("Visualisasi Performa Model LSTM")
        st.write("Grafik ini merepresentasikan bagaimana model belajar selama proses training dan bagaimana performanya dalam memprediksi data uji.")

        # --- Data History Training (diekstrak dari output notebook Anda) ---
        st.subheader("Kurva Pembelajaran (Learning Curves)")
        history_data = {
            'epoch': list(range(1, 21)),
            'accuracy': [0.6159, 0.8797, 0.9504, 0.9691, 0.9802, 0.9837, 0.9877, 0.9924, 0.9924, 0.9965, 0.9965, 0.9930, 0.9977, 0.9965, 0.9965, 0.9936, 0.9883, 0.9959, 0.9947, 0.9977],
            'val_accuracy': [0.6515, 0.8004, 0.8914, 0.8949, 0.8984, 0.9422, 0.9527, 0.9089, 0.9299, 0.9282, 0.8949, 0.9440, 0.9685, 0.9650, 0.9527, 0.9545, 0.9650, 0.9632, 0.9685, 0.9737],
            'loss': [2.9279, 1.7019, 1.0429, 0.6926, 0.4642, 0.3325, 0.2375, 0.1804, 0.1441, 0.1114, 0.0925, 0.0888, 0.0730, 0.0703, 0.0584, 0.0631, 0.0758, 0.0555, 0.0500, 0.0423],
            'val_loss': [2.6640, 1.9057, 1.4449, 1.1596, 0.9614, 0.7597, 0.6282, 0.5664, 0.4784, 0.3807, 0.3535, 0.2621, 0.1929, 0.1808, 0.1924, 0.2115, 0.1663, 0.1881, 0.1682, 0.1785]
        }
        history_df = pd.DataFrame(history_data)

        col1, col2 = st.columns(2)
        with col1:
            fig_acc = px.line(history_df, x='epoch', y=['accuracy', 'val_accuracy'], title="Model Accuracy", markers=True)
            st.plotly_chart(fig_acc, use_container_width=True)
        with col2:
            fig_loss = px.line(history_df, x='epoch', y=['loss', 'val_loss'], title="Model Loss", markers=True)
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # --- Confusion Matrix (diekstrak dari output notebook Anda) ---
        st.subheader("Confusion Matrix")
        st.write("Menunjukkan seberapa baik model dapat membedakan antar kelas. Sumbu Y adalah label asli (True Label), dan sumbu X adalah label prediksi (Predicted Label).")

        # Data cm dari output cell 39 di notebook.
        cm_data = np.array([[279, 1, 4],
                            [1, 166, 1],
                            [12, 2, 105]])
        
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Normal", "Penipuan", "Promo"],
                    yticklabels=["Normal", "Penipuan", "Promo"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig_cm)