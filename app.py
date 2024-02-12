import joblib
import streamlit as st
import numpy as np
import pandas as pd

model2 = joblib.load('model2.joblib')

st.title('House Price Prediction :house:')
size = st.number_input('House Size (Sq. Ft.)', 100)
bedrooms = st.slider('Number of Bedrooms', 1, 5)
bathrooms = st.slider('Number of Bathrooms', 1, 5)
year = st.number_input('Year Built', 2000)
distance = st.slider('Distance from City Center', 1, 20)


columns = ['House_Size', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Distance_to_City_Center']


def predict():
     row =  np.array([size, bedrooms, bathrooms, year, distance])
     x = pd.DataFrame([row], columns=columns)
     prediction = model2.predict(x)
     st.text(prediction)


st.button('Predict Price', on_click = predict)
