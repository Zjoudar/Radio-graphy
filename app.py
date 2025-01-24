import os
import uuid
import flask
import urllib
from PIL import Image
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np 
import logging
from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import os
from tensorflow.keras.models import load_model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
global model
model = load_model(os.path.join(BASE_DIR, 'Caries_Detection_128.h5'))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('IndexApp.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('IndexApp.html', prediction='No image uploaded!')

    imageFile = request.files['imagefile']

    if imageFile.filename == '':
        return render_template('IndexApp.html', prediction='No selected file')

    if imageFile and allowed_file(imageFile.filename):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imageFile.filename)
        imageFile.save(image_path)

        try:
            # Handle potential grayscale images (assuming 3 channels for ResNet50)
            image = load_img(image_path, target_size=(128, 128))
            if image.mode == 'L':  # Grayscale
                image = image.convert('RGB')

            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)  # Batch for potential efficiency in model.predict

            image = preprocess_input(image)
            yhat = model.predict(image)

            # Handle model output format (assuming single class label)
            predicted_class_index = np.argmax(yhat)
            predicted_class = "Predicted Class: " + str(predicted_class_index)  # Replace with your class labels if needed

            return render_template('IndexApp.html', prediction=predicted_class)

        except Exception as e:
            return render_template('IndexApp.html', prediction=f"Error processing image: {str(e)}")

        else:
            return render_template('IndexApp.html', prediction='Invalid image format')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'  # Assuming you have an 'uploads' directory for storing images
    app.run(port=3000, debug=True)
