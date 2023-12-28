import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import json

class Operations:
    def __init__(self):
        self.model = None
        self.class_names = None        

    def select_model(self, plant_name):
        with open('config/final_model_path.json', 'r') as file:
            self.model_mapping = json.load(file)
        if plant_name not in self.model_mapping:
            error_msg = "Invalid plant name. Choose among Potato, Pepper, or Tomato."
            raise ValueError(error_msg)

        
        model_info = self.model_mapping[plant_name]
        self.model = load_model(model_info["model_path"])
        self.class_names = model_info["class_names"]

        return self.model, self.class_names
    
    
    def _treatment_for_disease(self, plant_name, predicted_class):
        with open('config/treatments.json', 'r') as file:
            treatments = json.load(file)
        return treatments.get(plant_name, {}).get(predicted_class, "Treatment not found for this disease.")

    def make_prediction(self, plant_name, model, class_names, img):
            try:
                # Preprocess the image
                target_size = (256, 256)  # Replace with your desired target size
                img = keras_image.array_to_img(img)
                resized_img = img.resize(target_size)
                resized_img_array = keras_image.img_to_array(resized_img)
                resized_img_array = np.expand_dims(resized_img_array, axis=0)  # Add batch dimension
                preprocessed_img = resized_img_array / 255.0  # Normalize pixel values
            
                # Make prediction
                predictions = model.predict(preprocessed_img)
            
                # Get predicted class and confidence
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = round(100 * np.max(predictions[0]), 2)
                treatment = self._treatment_for_disease(plant_name,predicted_class)

                return {
                    'class': predicted_class,
                    'confidence': float(confidence),
                    'treatment': treatment
                }
            
            except Exception as e:
                error_msg = f"Error making prediction: {str(e)}"
                return {"error": error_msg}
            
            
            
