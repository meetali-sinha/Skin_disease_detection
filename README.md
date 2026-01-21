==================================================
Skin Disease Detection Using Deep Learning
==================================================

Introduction
--------------------------------------------------
Skin diseases are common health conditions, and early detection is essential for timely treatment and prevention of complications.  
This project presents a web based Skin Disease Detection system developed using Deep Learning and Flask.

The application allows users to upload an image of a skin lesion and predicts the type of skin disease along with a confidence score, precautions, and brief medical information.

This system is created for educational and research purposes and does not replace professional medical diagnosis.

--------------------------------------------------
Objectives
--------------------------------------------------
To develop an automated skin disease detection system using deep learning  
To classify skin diseases from uploaded images  
To provide confidence scores for predictions  
To display disease related precautions and additional information  
To design a simple and user friendly web interface  

--------------------------------------------------
Key Features
--------------------------------------------------
Image based skin disease prediction  
Support for seven skin disease categories  
Displays confidence level for predictions  
Provides precautions and disease description  
Fast and efficient prediction using a trained CNN model  

--------------------------------------------------
Skin Diseases Covered
--------------------------------------------------
Melanocytic nevi  
Melanoma  
Benign keratosis like lesions  
Basal cell carcinoma  
Actinic keratoses  
Vascular lesions  
Dermatofibroma  

--------------------------------------------------
Technology Stack
--------------------------------------------------
Frontend  
HTML, CSS  

Backend  
Flask, Python  

Deep Learning  
TensorFlow, Keras  

Image Processing  
NumPy  

Model Architecture  
Convolutional Neural Network  

--------------------------------------------------
Project Structure
--------------------------------------------------
SkinDiseaseDetection  
│  
├── app.py  
├── model.h5  
├── requirements.txt  
├── README.md  
│  
├── templates  
│   └── base.html  
│  
├── static  
│   ├── css  
│   └── images  
│  
└── uploads  
    └── .gitkeep  

--------------------------------------------------
How to Run the Project
--------------------------------------------------
Step 1 Clone the repository  
git clone https://github.com/your-username/SkinDiseaseDetection.git  

Step 2 Navigate to the project directory  
cd SkinDiseaseDetection  

Step 3 Install required dependencies  
pip install -r requirements.txt  

Step 4 Run the Flask application  
python app.py  

Step 5 Open the application in browser  
http://127.0.0.1:5000  

--------------------------------------------------
Model Information
--------------------------------------------------
The deep learning model is trained using a Convolutional Neural Network on skin lesion images.  
The trained model is stored in h5 format and loaded without optimizer to ensure smooth inference during prediction.

--------------------------------------------------
Disclaimer
--------------------------------------------------
This project is developed for educational and research purposes only.  
It should not be considered as a medical diagnostic tool.  
Always consult a qualified dermatologist for medical advice.

--------------------------------------------------
Future Enhancements
--------------------------------------------------
Increase dataset size to improve accuracy  
Apply advanced image preprocessing and augmentation  
Deploy the application on cloud platforms  
Improve user interface and responsiveness  
Add user authentication and prediction history  

--------------------------------------------------
Author
--------------------------------------------------
Mitali Sinha  
Bachelor of Technology in Artificial Intelligence and Machine Learning  
Aspiring Data Analyst and Machine Learning Engineer  

--------------------------------------------------
Acknowledgement
--------------------------------------------------
This project was developed as part of academic learning to understand and apply deep learning concepts in healthcare applications.
==================================================
