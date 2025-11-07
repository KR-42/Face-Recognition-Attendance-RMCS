**Face Recognition Attendance System (RMCS)**

A smart attendance system that uses face recognition technology to automatically detect and record student presence in real-time using a webcam.
This project leverages TensorFlow, Keras-Facenet, and MTCNN for facial feature extraction and matching.

**Features:**

- Real-time face detection and recognition
- Automatic attendance logging in Excel
- Voice feedback using pyttsx3
- User-friendly UI with ttkbootstrap
- Easy to configure and extend


**how to run?**

1️. Clone the Repository

git clone https://github.com/KR-42/Face-Recognition-Attendance-RMCS.git

cd Face-Recognition-Attendance-RMCS

2️. Install Required Dependencies

pip install opencv-python numpy Pillow pandas openpyxl pyttsx3 matplotlib ttkbootstrap keras-facenet mtcnn scipy tensorflow


3. Run app

python recognize_and_attend.py



**Admin Menu – Register New Face**
To register a new face, follow these steps:


1. Run the application using the command above.

2. On the main interface, go to the Admin Menu.

3. Enter the admin password: admin123.

4. You will then be able to register new faces by capturing images through the webcam.

5. Once registered, the new faces will be automatically recognized in attendance mode.

⚠️ ***Make sure the lighting is good and the face is clearly visible when registering.***

