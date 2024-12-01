import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import uvicorn
app=FastAPI()
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import requests
import gdown
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def ping():
    return "Hello, I am alive"



MODEL_PATH = "./model.keras"
MODEL_URL = "https://drive.google.com/uc?id=16rZUeiO443-2ENWqTRyZaMiyJiomJp-e"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")

if os.path.exists(MODEL_PATH):
    print("File successfully downloaded. Loading model...")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
else:
    print("Download failed. File not found.")
CLASS_NAMES=[
    'Eczema 1677',
    'Warts Molluscum and other Viral Infections - 2103',
    'Melanoma 15.75k',
    'Atopic Dermatitis - 1.25k',
    'Basal Cell Carcinoma (BCC) 3323',
    'Melanocytic Nevi (NV) - 7970',
    'Benign Keratosis-like Lesions (BKL) 2624',
    'Psoriasis pictures Lichen Planus and related diseases - 2k',
    'Seborrheic Keratoses and other Benign Tumors - 1.8k',
    'Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k',
    'Monkeypox',
    'Normal'
]
def read_file_as_image(data) -> np.ndarray:
    # Read image file and preprocess
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")  # Ensure 3 color channels
    image = image.resize((224, 224))  # Resize to (224, 224)
    image_array = img_to_array(image)  # Convert to NumPy array
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions for batch
    processed_image = preprocess_input(image_array)  # Preprocess for ResNet50
    return processed_image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the uploaded image
    image = read_file_as_image(await file.read())

    # Perform prediction
    predictions = MODEL.predict(image)
    result_ind = int(np.argmax(predictions[0]))  # Get the index of the predicted class
    predicted_class = str(CLASS_NAMES[result_ind])  # Map index to class name
    confidence = float(np.max(predictions[0]))  # Confidence of the prediction

    return {
        "class": predicted_class,
        "confidence": confidence
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)