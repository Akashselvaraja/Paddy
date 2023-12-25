import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import gdown  # To download the model from Google Drive

# Defining the number of classes
num_classes = 5

# Defining the class labels
class_labels = {
    0: 'Brown Spot',
    1: 'Healthy',
    2: 'Hispa',
    3: 'Leaf Blast',
    4: 'Tungro'
}




model = load_model(r"C:\Users\abhis\Downloads\paddy\weight.h5")

# Function to preprocess the input image
def preprocess_image(img):
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension as the model expects it
    return img

# Streamlit app
def main():
    st.title("Leaf Disease Prediction App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_display = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)

        # Preprocess the uploaded image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img = preprocess_image(img)

        # Extract features using the VGG16 base model
        base_model = VGG16(weights='imagenet', include_top=False)
        img_features = base_model.predict(img)

        # Reshape the features to match the input shape of the student model
        img_features = img_features.reshape(1, 7, 7, 512)

        # Make predictions using the student model
        predictions = model.predict(img_features)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])

        # Map the index back to class label
        predicted_class_label = class_labels[predicted_class_index]

        # Display the prediction
        st.write(f"Predicted Class Label: {predicted_class_label}")

if __name__ == "__main__":
    main()
