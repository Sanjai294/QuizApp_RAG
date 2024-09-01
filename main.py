import streamlit as st
import openai
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def modify_image(image, prompt):
    # Convert image to RGBA format if it's not already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Save image to BytesIO
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    try:
        # Call OpenAI API to create an edited image
        response = openai.Image.create_edit(
            image=img_byte_arr,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        # Print response for debugging
        st.write(response)
        
        # Retrieve and return the modified image
        image_url = response['data'][0]['url']
        modified_image = Image.open(BytesIO(requests.get(image_url).content))
        return modified_image
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit App Interface
st.title("Face Modifier AI")
st.write("Upload an image and enter a prompt to modify it using OpenAI's API.")

# Upload image file
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Prompt for image modification
prompt = st.text_input("Enter a prompt to modify the image")

if st.button("Modify Image"):
    if uploaded_image is not None and prompt:
        with st.spinner("Modifying image..."):
            image = Image.open(uploaded_image)
            modified_image = modify_image(image, prompt)
            if modified_image:
                st.image(modified_image, caption='Modified Image', use_column_width=True)
                modified_image.save("modified_image.png")
                st.success("Modified image saved as 'modified_image.png'.")
                st.download_button("Download Image", data=open("modified_image.png", "rb").read(), file_name="modified_image.png")
            else:
                st.error("Failed to modify the image.")
    else:
        st.error("Please upload an image and enter a prompt.")
