from flask import Flask, render_template, request, flash, redirect, url_for
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Ensure TensorFlow 1.x code is compatible in a TensorFlow 2.x environment.
tf.compat.v1.disable_eager_execution()



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    print("Image shape:", img_array.shape)
    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] > 0.5)
    label = 'Uninfected' if predicted_class == 1 else 'Infected'
    return label, prediction[0][0]

def make_prediction(values, dic):
    try:
        values = np.asarray(values)
        model_path = ''
        
        if len(values) == 8:
            model_path = 'models/diabetes.pkl'
        elif len(values) == 26:
            model_path = 'models/breast_cancer.pkl'
        elif len(values) == 10:
            model_path = 'models/liver.pkl'
        else:
            raise ValueError(f"Invalid input length: {len(values)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            return model.predict(values.reshape(1, -1))[0]
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None
    
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print("Form Data Received:", to_predict_dict)
            print("List of Values:", to_predict_list)
            pred = predict(to_predict_list, to_predict_dict)
            print("Prediction Result:", pred)
    except Exception as e:
        print("Error:", e)
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)


@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    pred = None

    if request.method == 'POST':
        try:
            if 'image' in request.files:
                # Open the image and ensure it's grayscale
                img = Image.open(request.files['image']).convert('L')

                # Resize the image to match the model input (36x36)
                img = img.resize((36, 36))

                # Convert the image to a NumPy array and normalize it
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))  # Adding batch dimension

                # Ensure model is loaded once
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
            else:
                return render_template('pneumonia.html', message="Please upload an Image")

        except Exception as e:
            print(f"Prediction Error: {e}")
            return render_template('pneumonia.html', message="Error during prediction.")

    return render_template('pneumonia_predict.html', pred=pred)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)