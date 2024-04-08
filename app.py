import streamlit as st
from keras.preprocessing import image
import numpy as np
import cv2
from tensorflow.keras.utils import img_to_array
import os
import pickle

# Load the pre-trained model
model_pkl_file = "Face_classifier.pkl"
with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)

# Class labels
class_labels = {
    0: 'Ahegao',
    1: 'Angry',
    2: 'Happy',
    3: 'Neutral',
    4: 'Sad',
    5: 'Surprise'
}

# Function to detect faces and predict emotions
def detect_emotion(image):
    # Load Haar cascade classifier for face detection
    face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


    # Detect faces in the image
    faces = face_classifier.detectMultiScale(image, 1.3, 5)

    # Process each face
    for (x,y,w,h) in faces:
        # Extract face ROI
        face_roi = image[y:y+h, x:x+w]

        # Resize face ROI to the required input size of the model
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Make prediction using the model
        preds = model.predict(face_roi)[0]
        label = class_labels[preds.argmax()]

        # Draw bounding box and label on the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image

# Streamlit UI
def main():
    st.title("Facial Expression Detection")
    st.markdown("Upload an image to detect facial expression")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Detect emotions
        output_image = detect_emotion(image)

        # Display the output image
        st.image(output_image, channels="BGR", caption="Output Image", use_column_width=True)

if __name__ == "__main__":
    main()
