import streamlit as st
from PIL import Image

# --- JUDUL HALAMAN ---
st.title("🚀 My Projects")
st.write("Berikut adalah beberapa proyek yang pernah saya kerjakan:")
st.write("---")


# --- FUNGSI UNTUK MENAMPILKAN PROYEK ---
def project_card(image_url, title, description, link):
    """untuk menampilkan kartu proyek yang rapi."""

    with st.container():
        try:
            project_image = Image.open(requests.get(image_url, stream=True).raw)
        except:
            project_image = "https://via.placeholder.com/300x200.png?text=Project+Image"

        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(project_image, use_column_width=True)

        with col2:
            st.subheader(title)
            st.write(description)
            st.markdown(f"[🔗 Lihat Proyek disini]({link})", unsafe_allow_html=True)


# --- DAFTAR PROYEK ---

# --- PROYEK 1 ---
with st.container(border=True): 
    project_card(
        image_url="URL_GAMBAR_PROYEK_1",  
        title="FraudGuard App",
        description="""
        Working on final capstone project on Fraud Detection SMS/Email. “FraudGuard” is an application used to identify fraud. 
        FraudGuard has several main features, namely that users can input text from suspicious sms and suspicious emails and 
        then can see the output whether the text is included in fraud or not. This application also has an educational feature in the 
        form of knowledge articles about fraud that can be used as learning material for users..
        """,
        link="https://github.com/Bangkit-Team-C241-PS499/ML-FraudModel/tree/master" 
    )

st.write("") 

# --- PROYEK 2 ---
with st.container(border=True):
    project_card(
        image_url="URL_GAMBAR_PROYEK_2",
        title="Bengkod App",
        description="""
        Creating an engaging user interface. Delivering an experience that meets user needsDesigning the workflow for the attendance application. Developing the concept for an attendance application. Creating features for Class List, Class Details, Permission Form, Permission and Attendance History.
        """,
        link="https://play.google.com/store/apps/details?id=com.bengkelkoding.bengkel_koding_mobile&hl=en_US" 
    )

st.write("") # Memberi spasi

# --- PROYEK 3 ---
with st.container(border=True):
    project_card(
        image_url="URL_GAMBAR_PROYEK_3",
        title="NLP, Spotify Sentimen Analysis",
        description="""
        Developed a sentiment analysis system to categorize reviews on Spotify as positive, neutral, 
        or negative. Successfully tested the model's abilit to predict sentiment based on input provided. 
        Implemented the Logistic Regression algorithm for sentiment analysis
        """,
        link="https://github.com/yourusername/klasifikasi-iris-cnn" 
    )
