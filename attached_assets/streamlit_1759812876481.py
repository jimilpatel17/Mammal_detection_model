# import streamlit as st
# from PIL import Image
# import io
# import os
# import google.generativeai as genai

# # --- Configure Gemini ---
# api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCLII-7Lzjuq02PUvIufcY7PXep0sjX8fA")
# genai.configure(api_key=api_key)

# model = genai.GenerativeModel("gemini-2.0-flash-lite")

# # --- Streamlit App UI ---
# st.set_page_config(
#     page_title="Indian Mammal Species Identifier",
#     layout="centered",
#     page_icon="🦌"
# )

# st.title("🐾 Indian Mammal Species Identifier")
# st.markdown(
#     """
#     Upload a clear image of a **Mammal** from the **Indian subcontinent** (India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan).  
#     The AI Model will identify the mammal name .
#     """
# )

# uploaded_image = st.file_uploader("Upload Mammal Image (JPEG/PNG)", type=["jpg", "jpeg", "png", "HEIC", "HEIF"])

# # --- Inference Function ---
# def identify_mammal(image):
#     try:.
#         img = Image.open(image)

#         # Ensure RGB
#         if img.mode != "RGB":
#             img = img.convert("RGB")

#         buffered = io.BytesIO()
#         img.save(buffered, format="JPEG", quality=85)
#         img_bytes = buffered.getvalue()

#         prompt = """
# You are a wildlife expert focused on mammals of the Indian subcontinent. 
# Identify the species shown in the image, and provide the following information in a structured format:
# 1. **Scientific Name**: 

# **Important**: Only identify mammals native to or commonly found in the Indian subcontinent (India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan). If the image shows an animal that is not a mammal or not from this region, respond with "Not identifiable as a native mammal of the Indian subcontinent."
# """

#         response = model.generate_content(
#             contents=[{
#                 "parts": [
#                     {"text": prompt},
#                     {"inline_data": {
#                         "mime_type": "image/jpeg",
#                         "data": img_bytes
#                     }}
#                 ]
#             }],
#             stream=False
#         )

#         return response.text if response.text else "No response generated. Please try with a clearer image."

#     except Exception as e:
#         return f"❌ Error: {e}"

# # --- Run Inference ---
# if uploaded_image:
#     with st.spinner("Analyzing image..."):
#         result = identify_mammal(uploaded_image)
#         st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
#         st.subheader(" Species Information:")
#         st.text(result)

# # --- Tips Section ---
# st.markdown("---")
# with st.expander("💡 Tips for Best Results"):
#     st.markdown("""
# - Use clear, well-lit images
# - Make sure the animal is the main subject
# - Blurred or distant animals may not work well
# - Try cropping unnecessary background
# """)


# """
# Without using strealit UI :


# from PIL import Image
# import io
# import os
# import google.generativeai as genai

# # --- Configure Gemini ---
# api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCLII-7Lzjuq02PUvIufcY7PXep0sjX8fA")
# genai.configure(api_key=api_key)

# model = genai.GenerativeModel("gemini-2.0-flash-lite")

# # --- Inference Function ---
# def identify_mammal(image_path):
#     try:
#         img = Image.open(image_path)

#         # Ensure RGB
#         if img.mode != "RGB":
#             img = img.convert("RGB")

#         # Convert to bytes
#         buffered = io.BytesIO()
#         img.save(buffered, format="JPEG", quality=85)
#         img_bytes = buffered.getvalue()

#         # Prompt
#         prompt = """You are a wildlife expert focused on mammals of the Indian subcontinent.
# Identify the species shown in the image, and provide the following information in a structured format:
# 1. **Scientific Name**:"""

# **Important**: Only identify mammals native to or commonly found in the Indian subcontinent (India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan). 
# If the image shows an animal that is not a mammal or not from this region, respond with 
# "Not identifiable as a native mammal of the Indian subcontinent."
# """

#         # Send to Gemini
#         response = model.generate_content(
#             contents=[{
#                 "parts": [
#                     {"text": prompt},
#                     {"inline_data": {
#                         "mime_type": "image/jpeg",
#                         "data": img_bytes
#                     }}
#                 ]
#             }],
#             stream=False
#         )

#         return response.text if response.text else "No response generated. Please try with a clearer image."

#     except Exception as e:
#         return f"❌ Error: {e}"

# # --- Example Run ---
# if __name__ == "__main__":
#     test_image = "test_mammal.jpg"  # replace with your image path
#     result = identify_mammal(test_image)
#     print("\n Species Information:\n", result)

# """
import streamlit as st
from PIL import Image
import io
import os
import google.generativeai as genai

# --- Configure Gemini ---
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyD1SOK7JGlkh872ezZjnHCmtSNnfm1Ts5U")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash-lite")

# --- Streamlit App UI ---
st.set_page_config(
    page_title="Indian Mammal Species Identifier",
    layout="centered",
    page_icon="🦌"
)

st.title("🐾 Indian Mammal Species Identifier")
st.markdown(
    """
    Upload a clear image of a **Mammal** from the **Indian subcontinent** (India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan).  
    The AI Model will identify the mammal name.
    """
)

uploaded_image = st.file_uploader("Upload Mammal Image (JPEG/PNG)", type=["jpg", "jpeg", "png", "HEIC", "HEIF"])

# --- Inference Function ---
def identify_mammal(image):
    try:
        img = Image.open(image)

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()

        prompt = """You are a wildlife expert focused on mammals of the Indian subcontinent. 
Identify the species shown in the image, and provide the following information in a structured format:
1. **Scientific Name**

**Important**: Only identify mammals native to or commonly found in the Indian subcontinent (India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan). 
If the image shows an animal that is not a mammal or not from this region, respond with 
"Not identifiable as a native mammal of the Indian subcontinent."
"""

        response = model.generate_content(
            contents=[{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_bytes
                    }}
                ]
            }],
            stream=False
        )

        return response.text if response.text else "No response generated. Please try with a clearer image."

    except Exception as e:
        return f"❌ Error: {e}"

# --- Run Inference ---
if uploaded_image:
    with st.spinner("Analyzing image..."):
        result = identify_mammal(uploaded_image)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.subheader("Species Information:")
        st.text(result)

# --- Tips Section ---
st.markdown("---")
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
- Use clear, well-lit images  
- Make sure the animal is the main subject  
- Blurred or distant animals may not work well  
- Try cropping unnecessary background  
""")





# new api : AIzaSyD1SOK7JGlkh872ezZjnHCmtSNnfm1Ts5U