import streamlit as st
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle



knn = pickle.load(open('house_knn.pkl', 'rb'))
rf = pickle.load(open('house_rf.pkl', 'rb'))

st.subheader('House price prediction - King County, WA')

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.slider('Bedrooms', 1, 5, 2)
    living_sqft = st.slider('Living square footage', 800, 5000, 1000)


with col2:
    bathrooms = st.slider('Baths', 1, 5, 2)
    lot_size = st.slider('Lot size (sq. ft) ', 1000, 20000, 2000)

if lot_size < living_sqft:
    st.write('Select a lot size bigger than the house.')
else:
    knn_pred = knn.predict([[bedrooms, bathrooms, living_sqft, lot_size]])
    rf_pred = rf.predict([[bedrooms, bathrooms, living_sqft, lot_size]])

    st.markdown('---')
    st.subheader('Model predictions')
    st.write(f'KNN: ${knn_pred[0]:.0f}')
    st.write(f'Random Forest: ${rf_pred[0]:.0f}')
