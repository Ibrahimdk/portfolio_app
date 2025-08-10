import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from textblob import TextBlob
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- Judul Halaman ---
st.title("🤖 Implementasi Prediksi Model")
st.write("Ketik atau tempelkan satu atau beberapa pesan SMS di bawah ini, lalu klik tombol 'Prediksi'.")

# --- Kotak Input Teks ---
input_text = st.text_area("Masukkan Teks SMS di sini:", 
                          "Selamat anda mendapatkan hadiah undian sebesar 100jt dari kami. Silakan klik link berikut untuk info lebih lanjut.",
                          height=150)

# Tombol untuk memulai prediksi
if st.button("Prediksi", type="primary"):
    if input_text.strip() == "":
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Sedang memproses..."):
            # --- Memuat Model & Tokenizer (saat tombol ditekan) ---
            try:
                model = tf.keras.models.load_model('model.h5')
                with open('tokenizer.bin', 'rb') as f:
                    tokenizer = pickle.load(f)
                with open('slang.txt', "r", encoding="utf-8") as f:
                    slangs_raw = f.readlines()
                slangs = [re.split(r'[:]', s.strip('\n')) for s in slangs_raw]
                slang_dict = {key.strip(): val.strip() for key, val in slangs}
            except FileNotFoundError:
                st.error("Pastikan file 'model.h5', 'tokenizer.bin', dan 'slang.txt' ada di folder utama.")
                st.stop()
            
            # --- Definisikan fungsi preprocessing ---
            factory_stopword = StopWordRemoverFactory()
            stopword_remover = factory_stopword.create_stop_word_remover()
            stem_factory = StemmerFactory()
            stemmer = stem_factory.create_stemmer()

            def preprocess(text):
                text = text.lower()
                text = re.sub(r"\d+", "", text)
                text = text.translate(str.maketrans("", "", string.punctuation)).strip()
                text = re.sub(r'\s+', ' ', text)
                text = stopword_remover.remove(text)
                wordlist = TextBlob(text).words
                for i, v in enumerate(wordlist):
                    if v in slang_dict:
                        wordlist[i] = slang_dict[v]
                text = ' '.join(wordlist)
                text = stemmer.stem(text)
                return text

            # --- Pipeline Prediksi ---
            # 1. Pisahkan setiap baris teks jika ada lebih dari satu
            texts_to_predict = [line for line in input_text.split('\n') if line.strip() != ""]
            
            # 2. Preprocessing setiap teks
            cleaned_texts = [preprocess(txt) for txt in texts_to_predict]

            # 3. Tokenizing dan Padding
            sequences = tokenizer.texts_to_sequences(cleaned_texts)
            padded_sequences = pad_sequences(sequences, maxlen=120, padding='post')

            # 4. Prediksi
            predictions = model.predict(padded_sequences)
            predicted_classes = np.argmax(predictions, axis=1)

            # 5. Tampilkan Hasil
            label_map = {0: 'Normal', 1: 'Penipuan', 2: 'Promo'}
            
            st.success("✅ Prediksi Selesai!")
            st.subheader("Hasil Prediksi:")

            for i, text in enumerate(texts_to_predict):
                prediction_label = label_map[predicted_classes[i]]
                st.markdown(f"> **Teks:** `{text}`")
                
                if prediction_label == 'Penipuan':
                    st.error(f"**Prediksi: {prediction_label}** 🚨")
                elif prediction_label == 'Promo':
                    st.info(f"**Prediksi: {prediction_label}** 💰")
                else: # Normal
                    st.success(f"**Prediksi: {prediction_label}** 👍")
                st.write("---")