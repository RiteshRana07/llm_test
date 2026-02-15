import streamlit as st
import cv2
import requests
import numpy as np
from PIL import Image
from groq import Groq

# ---------------- CONFIG ----------------
st.set_page_config(page_title="FoodScan", layout="centered")

# Configure Gemini API from Streamlit Secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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

    nutriments = product.get('nutriments', {})  # FIXED BUG
    sugar = nutriments.get("sugars_100g", 0)
    salt = nutriments.get("salt_100g", 0)
    fat = nutriments.get("saturated-fat_100g", 0)

    ingredients = product.get("ingredients") or ""

    # Rule 0 (keep your logic intact)
    if not ingredients or ingredients.lower() in ["not available", "no ingredients present", "unknown"]:
        return "Decision: Not Recommended\nReason: Ingredients information missing"

    prompt = f"""
You are a health-based food recommendation assistant.

Your task is to analyze the user's health profile and the product's nutritional values, then decide whether the user should consume the product or avoid it.

User Health Profile:
- Diabetes: {user['diabetes']}
- Blood Pressure (BP): {user['bp']}
- Heart Condition: {user['heart']}
- Age: {user['age']}


Product Nutrition (per 100g):
- Sugar: {product['nutriments'].get('sugars_100g', 0)}
- Salt: {product['nutriments'].get('salt_100g', 0)}
- Saturated Fat: {product['nutriments'].get('saturated-fat_100g', 0)}

Decision Rules:
0. If ingredients is not present → Not Recommended
1. If the user has diabetes and sugar > 10 → Not Recommended
2. If the user has BP issues and salt > 0.6 → Not Recommended
3. If the user has a heart condition and saturated fat > 5 → Consume with Caution
4. If age < 5 and sugar is high or the food is junk → Not Recommended
5. If age is between 5–12 and sugar > 8 → Consume with Caution
6. If age > 60 and salt > 0.5 → Consume with Caution
7. If age > 60 and fat > 6 → Consume with Caution
8. If none of the above conditions apply → Recommended

Now, based on the rules above, generate a clear and personalized recommendation for the user.

Strictly follow this output format:

Decision: <text>
Reason: <Max 10 words, mention only the main health factor>
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You strictly follow health rules."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

 








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
    st.image(image, caption="Input Image", width="stretch")

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
