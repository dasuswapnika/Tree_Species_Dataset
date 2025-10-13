# app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

# -------------------------------
# Load saved models and data
# -------------------------------
@st.cache_data
def load_models():
    # Load recommender system
    tree_data = pd.read_pickle("tree_data.pkl")
    scaler = joblib.load("scaler.joblib")
    nn_model = joblib.load("nn_model.joblib")
    # Load CNN classifier
    cnn_model = load_model("basic_cnn_tree_species.h5")
    return tree_data, scaler, nn_model, cnn_model

tree_data, scaler, nn_model, cnn_model = load_models()

# -------------------------------
# Helper Functions
# -------------------------------
def recommend_trees(location_features):
    """Recommend tree species based on location/environment features"""
    scaled_features = scaler.transform([location_features])
    preds = nn_model.predict(scaled_features)
    top_indices = np.argsort(preds[0])[::-1][:5]
    return tree_data['species'].iloc[top_indices].values

def find_tree_locations(tree_name):
    """Return locations where the tree is found"""
    locations = tree_data[tree_data['species'] == tree_name][['latitude', 'longitude']]
    return locations

def classify_tree_image(img):
    """Classify tree species from an image"""
    img = img.resize((128, 128))  # Adjust size if your CNN expects different
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    preds = cnn_model.predict(img_array)
    class_idx = np.argmax(preds[0])
    species = tree_data['species'].unique()[class_idx]
    return species

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌳 Tree Intelligence Assistant", layout="wide")
st.title("🌳 Tree Intelligence Assistant")
st.markdown("""
This AI-powered app helps students and nature enthusiasts identify and explore tree species based on image, location, and tree attributes.
""")

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose Functionality",
    ("Recommend Trees by Location", "Find Locations for a Tree", "Identify Tree from Image")
)

# -------------------------------
# Recommend Trees by Location
# -------------------------------
if option == "Recommend Trees by Location":
    st.header("🌲 Recommend Trees by Location")
    st.write("Enter your location/environment features:")

    # Example location features
    soil_type = st.selectbox("Soil Type", ['Sandy', 'Clay', 'Loam'])
    temperature = st.slider("Average Temperature (°C)", 0, 50, 25)
    rainfall = st.slider("Annual Rainfall (mm)", 0, 4000, 1000)

    if st.button("Recommend Trees"):
        soil_map = {'Sandy': 0, 'Clay': 1, 'Loam': 2}  # Match your model encoding
        features = [soil_map[soil_type], temperature, rainfall]
        recommendations = recommend_trees(features)
        st.success("Top Recommended Trees:")
        for t in recommendations:
            st.write(f"🌳 {t}")

# -------------------------------
# Find Locations for a Tree
# -------------------------------
elif option == "Find Locations for a Tree":
    st.header("📍 Find Locations for a Tree")
    tree_name = st.selectbox("Select Tree Species", tree_data['species'].unique())
    if st.button("Show Locations"):
        locations = find_tree_locations(tree_name)
        if not locations.empty:
            st.map(locations)
        else:
            st.warning("No location data available for this species.")

# -------------------------------
# Identify Tree from Image
# -------------------------------
elif option == "Identify Tree from Image":
    st.header("📷 Identify Tree from Image")
    uploaded_file = st.file_uploader("Upload an image of the tree", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Tree Image', use_column_width=True)
        if st.button("Identify Tree"):
            try:
                species = classify_tree_image(image)
                st.success(f"Identified Tree Species: 🌳 {species}")
            except Exception as e:
                st.error(f"Error identifying tree: {e}")
