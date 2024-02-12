from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from twilio.rest import Client

app = FastAPI()

# CORS settings
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

# Load your TensorFlow model
MODEL = tf.keras.models.load_model("C:\\Users\\SHREY SHUKLA\\Downloads\\Desktop\\models\\2")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Twilio settings
TWILIO_ACCOUNT_SID = "AC7ac1dfff00931bbe008dcd6775645851"  # Your Account SID here
TWILIO_AUTH_TOKEN = "363ec68cdf13b81a47dc0a4b374720af"  # Your Auth Token here
TWILIO_WHATSAPP_NUMBER = "+916351326710"  # Replace with your Twilio WhatsApp number
TWILIO_DESTINATION_NUMBER = "+14155238886"  # Replace with your destination WhatsApp number
 # Replace with your destination WhatsApp number

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Function to read file as image
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize the image to have shape (256, 256, 3)






    image = image.resize((256, 256))
    image = np.array(image)
    return image

# Function to send WhatsApp message
def send_whatsapp_message(message):
    client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        body=message,
        to=TWILIO_DESTINATION_NUMBER
    )

# Endpoint for prediction
@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Send prediction result via WhatsApp
    message = f"Prediction: {predicted_class}, Confidence: {confidence}"
    background_tasks.add_task(send_whatsapp_message, message)

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
