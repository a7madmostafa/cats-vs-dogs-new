
import streamlit as st
import requests
from io import BytesIO
from helper import preprocess_image, load_my_model, predict_img


st.title("Cats and Dogs Classification App")

# Load the model
my_model = load_my_model('inception_model.keras')


# Choose to upload an image or put an image URL
option = st.selectbox(
    'How would you like to be contacted?',
    ('Upload an image', 'Enter an image URL'))

img = None

if option == 'Upload an image':
    # Upload an image
    img = st.file_uploader("Upload an image for cat or dog", type=["jpg", "png", "jpeg"])

else:
    # Enter an image URL
    url = st.text_input("Enter an image URL for cat or dog")
    if url:
        response = requests.get(url)
        # convert image to Binary
        img = BytesIO(response.content)

if img is not None:
    # Preprocess the image
    img_prep = preprocess_image(img)

    # Make prediction
    class_name = predict_img(img_prep, my_model)

    # Display the image with the prediction
    st.image(img, caption=f"Predicted : {class_name}", width=300, use_column_width=True)
   
 
