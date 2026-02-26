import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Load model yang sudah dibuat di Colab tadi
model = joblib.load('model_iris.pkl')
iris = load_iris()

st.title("Prediksi Spesies Bunga Iris")
st.write("Masukkan karakteristik bunga di bawah ini:")

# Membuat input sidebar untuk user
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Parameter Input')
st.write(df)

# Prediksi
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Hasil Prediksi')
st.write(iris.target_names[prediction][0])

st.subheader('Probabilitas Prediksi')
st.write(prediction_proba)
