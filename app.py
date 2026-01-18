# app.py
import streamlit as st

# -------------------------------
# Helper Function (RULE-BASED)
# -------------------------------
def recommend_trees(soil, temperature, rainfall):
    recommendations = []

    if soil == "Sandy":
        recommendations += ["Coconut", "Casuarina"]
    if soil == "Clay":
        recommendations += ["Teak", "Banyan"]
    if soil == "Loam":
        recommendations += ["Neem", "Peepal"]

    if rainfall > 800:
        recommendations.append("Peepal")
    if 20 <= temperature <= 35:
        recommendations.append("Neem")

    return list(set(recommendations))


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌳 Tree Intelligence Assistant", layout="wide")
st.title("🌳 Tree Intelligence Assistant")
st.markdown("""
This AI-powered app helps students and nature enthusiasts identify and explore tree species 
based on environmental conditions.
""")

option = st.sidebar.selectbox(
    "Choose Functionality",
    (
        "Recommend Trees by Location",
        "Find Locations for a Tree",
        "Identify Tree from Image"
    )
)

# -------------------------------
# Recommend Trees by Location
# -------------------------------
if option == "Recommend Trees by Location":
    st.header("🌲 Recommend Trees by Location")

    soil_type = st.selectbox("Soil Type", ["Sandy", "Clay", "Loam"])
    temperature = st.slider("Average Temperature (°C)", 0, 50, 25)
    rainfall = st.slider("Annual Rainfall (mm)", 0, 4000, 1000)

    if st.button("Recommend Trees"):
        recommendations = recommend_trees(soil_type, temperature, rainfall)
        st.success("Top Recommended Trees:")
        for tree in recommendations:
            st.write(f"🌳 {tree}")

# -------------------------------
# Other Options (Disabled Safely)
# -------------------------------
elif option == "Find Locations for a Tree":
    st.warning("📍 Location data model not included in this demo.")

elif option == "Identify Tree from Image":
    st.warning("📷 Image classification model not included in this demo.")
