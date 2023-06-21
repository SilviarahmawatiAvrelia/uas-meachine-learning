import streamlit as st
from PIL import Image
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.models import load_model
from io import BytesIO


st.set_page_config(
    page_title="Project"
)
st.title('Classification Of Rice Leaves')

tab1, tab2 = st.tabs(["Information", "Test Image"])

with tab1:
    st.subheader("Pada klasifikasi ini terbagi menjadi 4 kelas atau label yaitu :")
    st.write("""
    <ol>
        <li>Brown Spot : Penyakit bercak daun coklat pada tanaman padi yaitu Oryzae berwarna coklat, bersekat 6-17, berbentuk silindris, agak melengkung, dan bagian tengahnya agak melebar.</li>
        <li>Hispa: penyakit yang memiliki bercak putih besar akibat serangan serangga dewasa yang mengikis permukaan daun.</li>
        <li>Leaf Blast: penyakit yang memiliki bercak kuning pada bagian ujung, hingga berwarna kecoklatan dan juga kering pada tanaman.</li>
        <li>Healthy : Memiliki warna hijau cerah, bentuk yang khas, permukaan yang halus, dan struktur pembuluh daun yang terlihat jelas.</li>
    </ol>""",unsafe_allow_html=True)
with tab2:
        model = load_model('cnn_model.h5')
        classes =["_BrownSpot","_Hispa","_LeafBlast","_Healthy"]
        # Menu pilihan
        menu = st.selectbox("Capture Option :",["Upload Photo", "Camera"])

        if menu == "Upload Photo":
            uploaded_file = st.file_uploader("Select photo", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Photo', use_column_width=True)
                # Mengubah gambar menjadi bentuk yang sesuai untuk prediksi
                resized_image = image.resize((128, 128))
                processed_image = np.array(resized_image) / 255.0
                input_image = np.expand_dims(processed_image, axis=0)

                # Melakukan prediksi menggunakan model atau tindakan lain
                prediction = model.predict(input_image)
                class_index = np.argmax(prediction[0])
                class_name = classes[class_index]

                # Menampilkan hasil prediksi
                st.success(f"Hasil Prediksi: {class_name}")

        elif menu == "Camera":
            st.write("Click the camera button below.")

            if st.button('Camera'):
                cap = cv2.VideoCapture(0)  # Menggunakan kamera utama

                ret, frame = cap.read()  # Membaca frame pertama dari kamera

                if ret:
                    st.image(frame, channels="BGR")

                # Mengubah gambar menjadi bentuk yang sesuai untuk prediksi
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(128, 128))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)

                # Melakukan prediksi menggunakan model atau tindakan lain
                prediction = model.predict(img)
                class_index = np.argmax(prediction[0])
                class_name = classes[class_index]

                # Menampilkan hasil prediksi
                st.success(f"Hasil Prediksi: {class_name}")

                # Menampilkan gambar hasil prediksi
                st.image(img[0], channels="RGB", caption='Predicted Image', use_column_width=True)