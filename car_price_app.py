import streamlit as st
import pandas as pd
import joblib

st.title("Bilpris-prediktion")

# Ladda modell
model = joblib.load("car_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Dela upp UI i sektioner
st.subheader("Grundinformation")
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Märke", ["Kia", "Chevrolet", "Mercedes", "Audi", "Volkswagen",
                                  "Toyota", "Honda", "BMW", "Ford", "Hyundai"])
    fuel_type = st.selectbox("Bränsletyp", ["Diesel", "Hybrid", "Electric", "Petrol"])

with col2:
    transmission = st.selectbox("Växellåda", ["Manual", "Automatic", "Semi-Automatic"])
    year = st.number_input("Årsmodell", 1990, 2025, 2015)

st.subheader("Tekniska detaljer")
engine_size = st.slider("Motorstorlek (L)", 0.5, 6.0, 2.0)
mileage = st.number_input("Miltal", 0, 500000, 10000)

st.subheader("Övrigt")
doors = st.selectbox("Antal dörrar", [2, 3, 4, 5])
owner_count = st.selectbox("Antal tidigare ägare", [1, 2, 3, 4, 5])

if st.button("Beräkna pris"):
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    # Numeriska värden
    input_data['year'] = year
    input_data['engine_size'] = engine_size
    input_data['mileage'] = mileage
    input_data['doors'] = doors
    input_data['owner_count'] = owner_count

    # Dummy encoding
    brand_col = f"brand_{brand.lower()}"
    if brand_col in input_data.columns:
        input_data[brand_col] = 1

    fuel_col = f"fuel_type_{fuel_type.lower()}"
    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1

    trans_col = f"transmission_{transmission.lower()}"
    if trans_col in input_data.columns:
        input_data[trans_col] = 1

    # Prediktion
    prediction = model.predict(input_data)

    # Visa resultat
    st.success(f"Predikterat pris: **{prediction[0]:,.0f} kr**")
