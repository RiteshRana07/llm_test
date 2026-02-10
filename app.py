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
        "ingredients": product.get("ingredients_text", "Not available"),
        "sugar": safe_float(nutriments.get("sugars_100g")),
        "salt": safe_float(nutriments.get("salt_100g")),
        "fat": safe_float(nutriments.get("saturated-fat_100g")),
    }

def health_decision(user, product):
    prompt = f"""
You are a food health recommendation system.

User profile:
- Diabetes: {user['diabetes']}
- BP: {user['bp']}
- Heart disease: {user['heart']}
- Age: {user['age']}
- Diet: {user['diet']}

Food nutrition (per 100g):
- Sugar: {product['nutriments'].get('sugars_100g', 0)}
- Salt: {product['nutriments'].get('salt_100g', 0)}
- Saturated Fat: {product['nutriments'].get('saturated-fat_100g', 0)}

Rules:
0. If ingredients is not present -> Not Recommended
1. Diabetes AND sugar > 10 → Not Recommended
2. BP AND salt > 0.6 → Not Recommended
3. Heart disease AND fat > 5 → Consume with Caution
4. Age < 5 AND sugar > 8 → Not Recommended
5. Age 5–12 AND sugar > 8 → Consume with Caution
6. Age > 60 AND salt > 0.5 → Consume with Caution
7. Age > 60 AND fat > 6 → Consume with Caution
8. Else → Recommended
9. Sugar > 10 AND sugar ≤ 15 → Consume with Caution
10. Salt > 1.0 → Not Recommended
11. Salt > 0.6 AND salt ≤ 1.0 → Consume with Caution
12. Saturated fat > 10 → Not Recommended
13. Saturated fat > 5 AND fat ≤ 10 → Consume with Caution
14. Sugar > 10 AND salt > 0.6 → Not Recommended
15. Sugar > 10 AND fat > 5 → Not Recommended
16. Age > 60 AND sugar > 8 AND salt > 0.5 → Not Recommended
17. Age ≤ 12 AND (sugar > 12 OR salt > 0.8) → Not Recommended
18. Else → Recommended

Give a clear recommendation based on the Rules which is mentioned above and strictly go through and follow the Rules:
- Recommended / Consume with caution / Not recommended
- Explain why in simple language with 30 words.
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

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
