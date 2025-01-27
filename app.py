import os
import time
import cv2
import uuid
import flask
from flask import Flask, request,render_template, redirect,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
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

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('Home.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/welcome')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/welcome')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')

@app.route('/welcome', methods=['GET'])
def hello_world():
    return render_template('welcome.html')

@app.route('/working', methods=['GET'])
def working():
    return render_template('working.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('About.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/pricing', methods=['GET'])
def pricing():
    return render_template('pricing.html')
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
            #image = load_img(image_path, target_size=(128, 128))
            start_time = time.time()
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))
          #  if image.mode == 'L':  # Grayscale
           #     image = image.convert('RGB')

            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)  # Batch for potential efficiency in model.predict

            image = preprocess_input(image)
            yhat = model.predict(image)

            # Handle model output format (assuming single class label)
            predicted_class_index = np.argmax(yhat)
            predicted_class = "Predicted Class: " + str(predicted_class_index)  # Replace with your class labels if needed
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Image processing time: {processing_time:.4f} seconds") 
            return render_template('IndexApp.html', prediction=predicted_class)

        except (IOError, OSError) as e:
                print(f"Error processing image: {str(e)}")
                return render_template('index.html', prediction="Error processing image")

    else:
         return render_template('IndexApp.html', prediction='Invalid image format')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'  # Assuming you have an 'uploads' directory for storing images
    app.run(port=3000, debug=True)
