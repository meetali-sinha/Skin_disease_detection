from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your Keras model
model = load_model('model.h5.keras')  # Replace 'my_model.h5' with your model path

# Dictionary mapping class indices to class names
lesion_classes_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}
precautions_dict = {
    'Melanocytic nevi': 'Regular skin examinations, avoid sun exposure, use sunscreen.',
    'Melanoma': 'Regular skin examinations, avoid sun exposure, use sunscreen, monitor moles for changes.',
    'Benign keratosis-like lesions': 'Regular skin examinations, avoid sun exposure, use sunscreen.',
    'Basal cell carcinoma': 'Regular skin examinations, avoid sun exposure, use sunscreen.',
    'Actinic keratoses': 'Regular skin examinations, avoid sun exposure, use sunscreen.',
    'Vascular lesions': 'Regular skin examinations, avoid sun exposure, use sunscreen.',
    'Dermatofibroma': 'Regular skin examinations, avoid sun exposure, use sunscreen.'
}

# Additional information for each class
additional_info_dict = {
    'Melanocytic nevi': 'Melanocytic nevi are benign skin growths and are usually harmless.',
    'Melanoma': 'Melanoma is a type of skin cancer that develops from melanocytes, the cells that produce melanin.',
    'Benign keratosis-like lesions': 'Benign keratosis-like lesions are non-cancerous skin growths.',
    'Basal cell carcinoma': 'Basal cell carcinoma is the most common type of skin cancer.',
    'Actinic keratoses': 'Actinic keratoses are precancerous growths that can develop into squamous cell carcinoma.',
    'Vascular lesions': 'Vascular lesions are abnormalities of blood vessels in the skin.',
    'Dermatofibroma': 'Dermatofibroma is a benign skin tumor.'
}
# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def model_predict(image_path, model):
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds)
    confidence_score = np.max(preds) * 100
    return pred_class_index, confidence_score

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return 'No selected file', 400

        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make prediction
            pred_class_index, confidence_score = model_predict(file_path, model)

            # Get the predicted class label
            predicted_class = lesion_classes_dict[pred_class_index]

            predicted_class = lesion_classes_dict[pred_class_index]
            precautions = precautions_dict.get(predicted_class, 'Precautions not available')
            additional_info = additional_info_dict.get(predicted_class, 'Additional information not available')

            # Format the prediction result
            result = f'{predicted_class}\n Confidence: {confidence_score:.2f}% \n Precautions:{precautions}{additional_info}'

            return result


    return 'Invalid request', 400

if __name__ == '__main__':
    app.run(debug=True)