import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.title("Rita en siffra")

# Tränar modellen på MNIST med RandomForest
@st.cache_resource
def train_model():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"][:10000], mnist["target"][:10000].astype(int)

    # Skalar datan
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tränar RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = train_model()

# Skapar ritcanvas
canvas = st_canvas(
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
)

# Förbereder bilden och gör prediktion
if canvas.image_data is not None and np.sum(canvas.image_data) > 0:
    img = canvas.image_data[:, :, 0]

    # Vänder färger så siffran blir vit på svart bakgrund
    img = 255 - img

    # Croppar bilden till det som faktiskt ritats
    coords = np.column_stack(np.where(img > 0))
    if len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Gör bilden kvadratisk och centrerar
    size = max(img.shape)
    square = np.zeros((size, size))
    x_offset = (size - img.shape[1]) // 2
    y_offset = (size - img.shape[0]) // 2
    square[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    img = cv2.resize(square, (28, 28))

    # Flatten och skala för modellen
    img_flat = img.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    # Prediktion
    pred = model.predict(img_scaled)
    st.write("Predikterad siffra:", pred[0])


