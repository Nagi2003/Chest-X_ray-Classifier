
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load the trained model
# Assuming your model_CNN.h5 file is loaded here
model = load_model('model_CNN.h5')

# Function to process image and make prediction
def predict_image(image):
    # Preprocess the image as needed (e.g., resize, normalize)
    resized_image = cv2.resize(image, (256, 256))
    processed_image = resized_image / 255.0  # Normalize pixel values
    processed_image = np.expand_dims(processed_image, axis=0)

    # Make prediction using your loaded model
    prediction = model.predict(processed_image)
    pneumonia_probability = prediction[0][0]  # Probability of pneumonia class

    # Determine the predicted label based on probability threshold
    if pneumonia_probability > 0.5:
        predicted_label = 'Pneumonia'
    else:
        predicted_label = 'Normal'

    return predicted_label, processed_image



# Route for uploading image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        # Read the image using OpenCV
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the uploaded image and make prediction
        predicted_label, processed_image = predict_image(image)

        # Convert image to base64
        retval, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        # Render HTML template with prediction result and image
        return render_template('result.html', prediction=predicted_label, image=base64_image)

    # Render HTML template for uploading file
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
