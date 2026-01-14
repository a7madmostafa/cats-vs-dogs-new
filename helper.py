
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import os

def preprocess_image(img):
    """
    Preprocesses an image by loading it, resizing it to a target size, converting it to an array, 
    expanding the dimensions of the array, and normalizing the pixel values.
    
    Parameters:
    img (str): The path to the image file.
    
    Returns:
    numpy.ndarray: The preprocessed image as a numpy array.
    """
    img = image.load_img(img, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255 
    return img

def load_my_model(model_file):
    """
    Load a pre-trained model from the given model path.

    Parameters:
        model_path (str): The path to the saved model.

    Returns:
        model: The loaded model.
    """
    # Get the absolute path to the current directory
    current_dir = os.path.dirname(__file__)

    # Construct the absolute path to your model file
    model_path = os.path.join(current_dir, model_file)

    model = load_model(model_path, compile=False)
    model.compile(RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_img(img, model):
    """
    Predicts the class of an image based on a given model.

    Parameters:
        img (numpy.ndarray): The input image to be predicted.
        model (tensorflow.python.keras.engine.training.Model): The trained model used for prediction.

    Returns:
        str: The predicted class of the image. It can be either 'Dog' or 'Cat'.
    """
    pred = model.predict(img)[0][0]
    class_name = 'Dog' if pred >= 0.5 else 'Cat'
    return class_name
