import streamlit as st
import cv2
import requests
import numpy as np
from PIL import Image
import google.generativeai as genai

# ---------------- CONFIG ----------------
st.set_page_config(page_title="FoodScan", layout="centered")

# Configure Gemini API from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ---------------- UTILS ----------------
def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0

def scan_barcode(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = cv2.barcode_BarcodeDetector()

    decoded_info, decoded_type, points = detector.detectAndDecode(gray)

    if decoded_info is None or decoded_info == "":
        return None

    # decoded_info may be list or string depending on OpenCV version
    if isinstance(decoded_info, list):
        return decoded_info[0]

    return decoded_info

def get_product_from_api(barcode):
    url = f"https://world.openfoodfacts.net/api/v2/product/{barcode}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None

    data = r.json()
    if data.get("status") != 1:
        return None

    product = data.get("product", {})
    nutriments = product.get("nutriments", {})
    
    return {
       "name": product.get("product_name", "Unknown"),
       "nutriments": product.get("nutriments", {}),
       "ingredients": product.get("ingredients_text", "No ingredients present"),
       "labels": product.get("labels", "")
   }

def health_decision(user, product):
    nutriments = product.get("nutriments", {})
    sugar = float(nutriments.get("sugars_100g", 0) or 0)
    salt = float(nutriments.get("salt_100g", 0) or 0)
    fat = float(nutriments.get("saturated-fat_100g", 0) or 0)
    ingredients = product.get("ingredients_text", "")

    # -------- STRICT RULE ENGINE (PRIMARY LOGIC) --------
    # Rule 0
    if not ingredients:
        decision = "Not Recommended"
        reason = "Ingredients information missing"

    # Disease rules (ONLY when user selects profile)
    elif user["diabetes"] and sugar > 10:
        decision = "Not Recommended"
        reason = "High sugar unsafe for diabetes"

    elif user["bp"] and salt > 0.6:
        decision = "Not Recommended"
        reason = "High salt unsafe for BP"

    elif user["heart"] and fat > 5:
        decision = "Consume with Caution"
        reason = "High fat for heart patients"

    # Age rules (always apply)
    elif user["age"] < 5 and sugar > 8:
        decision = "Not Recommended"
        reason = "Too much sugar for child"

    elif 5 <= user["age"] <= 12 and sugar > 8:
        decision = "Consume with Caution"
        reason = "High sugar for children"

    elif user["age"] > 60 and salt > 0.5:
        decision = "Consume with Caution"
        reason = "Salt not ideal for seniors"

    elif user["age"] > 60 and fat > 6:
        decision = "Consume with Caution"
        reason = "High fat for elderly"

    # General nutrition rules
    elif sugar > 10 and salt > 0.6:
        decision = "Not Recommended"
        reason = "High sugar and salt"

    elif sugar > 10 and fat > 5:
        decision = "Not Recommended"
        reason = "High sugar and fat"

    elif sugar > 15 or salt > 1.0 or fat > 10:
        decision = "Not Recommended"
        reason = "Very unhealthy nutrition levels"

    elif (10 < sugar <= 15) or (0.6 < salt <= 1.0) or (5 < fat <= 10):
        decision = "Consume with Caution"
        reason = "Moderate unhealthy nutrients"

    else:
        decision = "Recommended"
        reason = "Nutritionally safe choice"

    # -------- OPTIONAL: Gemini Explanation (your original prompt) --------
    prompt = f"""
You are a food health recommendation system.

User profile:
- Diabetes: {user['diabetes']}
- BP: {user['bp']}
- Heart disease: {user['heart']}
- Age: {user['age']}
- Diet: {user['diet']}

Food nutrition (per 100g):
- ingredients: {ingredients}
- Sugar: {sugar}
- Salt: {salt}
- Saturated Fat: {fat}

Final system decision: {decision}
Reason: {reason}

Explain why in simple language with 30 words.
"""

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return f"Decision: {decision}\nReason: {reason}\n\nExplanation: {response.text.strip()}"


# ---------------- UI ----------------
st.title("🥗 FoodScan – Smart Food Analyzer")

st.subheader("👤 Health Profile")
age = st.number_input("Age", 1, 120, 24)
diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian"])
diabetes = st.checkbox("Diabetes")
bp = st.checkbox("High Blood Pressure")
heart = st.checkbox("Heart Disease")

user_profile = {
    "age": age,
    "diet": diet,
    "diabetes": diabetes,
    "bp": bp,
    "heart": heart,
}

st.divider()

scan_mode = st.radio("Choose scan method", ["Upload Image", "Camera Scan"])
image = None

if scan_mode == "Upload Image":
    uploaded = st.file_uploader("Upload barcode image", ["jpg", "png", "jpeg"])
    if uploaded:
        img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.cvtColor(cv2.imdecode(img_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
else:
    cam = st.camera_input("Scan barcode using camera")
    if cam:
        image = np.array(Image.open(cam).convert("RGB"))

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    barcode = scan_barcode(image)
    if barcode:
        st.success(f"Barcode detected: {barcode}")

        product = get_product_from_api(barcode)
        if product:
            st.subheader("📦 Product Info")
            st.write("**Name:**", product["name"])
            st.write("**Ingredients:**", product["ingredients"])

            st.subheader("🧠 Health Recommendation")
            with st.spinner("Analyzing..."):
                result = health_decision(user_profile, product)

            st.info(result)
        else:
            st.error("Product not found in OpenFoodFacts.")
    else:
        st.error("No barcode detected. Try a clearer image.")
