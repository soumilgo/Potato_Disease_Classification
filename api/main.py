from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from fastapi.middleware.cors import CORSMiddleware

import requests

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# prod_model = TFSMLayer("../models/1", call_endpoint='serving_default')
# beta_model = TFSMLayer("../models/2", call_endpoint='serving_default')
endpoint = "http://localhost:8501/v1/models/email_model:predict"

#MODEL = TFSMLayer("../models/3", call_endpoint='serving_default')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        image = read_file_as_image(await file.read())
        
        image = tf.convert_to_tensor(image, dtype=tf.float32)
       
        image_batch = np.expand_dims(image, 0)
        
        json_data= {
            "instances": image_batch.tolist()
        }
        response = requests.post(endpoint, json=json_data)

        
        #return (response.json())
        prediction = np.array(response.json()["predictions"][0])

        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = round(np.max(prediction), 2)
        
        # predictions = MODEL(image_batch)  
        # predictions = predictions['output_0'].numpy()
        # predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        # confidence = round(100 * np.max(predictions[0]), 2)

       
        return {
            "class": predicted_class,
            "confidence": f"{confidence}%"
        }

    except Exception as e:
       
        return {"error": str(e)}
