import pandas as pd
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import tf_keras


# Load the trained model
model = tf.keras.models.load_model("model2")
print(model.summary())
def preprocess_input(core_temp, ambient_temp, ambient_humidity, bmi, heart_rate, age, skin_resistance):
    # Create a dictionary with input data
    data = {
        "coreTemp": [core_temp],
        "ambientTemp": [ambient_temp],
        "ambientHumidity": [ambient_humidity],
        "bmi": [bmi],
        "heartRate": [heart_rate],
        "age": [age],
        "skinRes": [int(skin_resistance)]  # Ensure skinRes is represented as int64
    }
    # Create a DataFrame from the dictionary
    input_df = pd.DataFrame(data)
    # Convert data types to match model's expectations (if necessary)
    input_df = input_df.astype({"coreTemp": "float32", "ambientTemp": "float32", "ambientHumidity": "float32",
                                 "bmi": "float32", "heartRate": "float32", "age": "float32", "skinRes": "int64"})
    return input_df

# Function to make predictions
def predict(input_df, model):
    # Convert input DataFrame to a TensorFlow dataset
    input_ds = tfdf.keras.pd_dataframe_to_tf_dataset(input_df)
    # Make predictions using the model
    predictions = model.predict(input_ds)
    return predictions

# Example input data
core_temp = 39
ambient_temp = 40.8
ambient_humidity = 0.4
bmi = 24
heart_rate = 166
age = 38
skin_resistance = 0

# Preprocess the input data
input_df = preprocess_input(core_temp, ambient_temp, ambient_humidity, bmi, heart_rate, age, skin_resistance)

# Make predictions
predictions = predict(input_df, model)

# Print the predictions
print("Predictions:")
print(predictions)


threshold = 0.5  # Threshold for binary classification

# Convert probability score to binary prediction
binary_prediction = 1 if predictions[0][0] > threshold else 0

# Print the binary prediction
print("Binary Prediction:", binary_prediction)