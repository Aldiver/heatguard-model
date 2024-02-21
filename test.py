import tensorflow as tf
import pandas as pd

# Load the saved TensorFlow Decision Forests model
loaded_model = tf.saved_model.load("tfdf_model")

# Function to get input from user
def get_input(feature_name):
    value = input(f"{feature_name} = ")
    return float(value)

# Guide display for user input
print("Enter data (Example)")
print("Core Temp = 39")
print("Ambient Temp = 40.8")
print("Ambient Humidity = 0.4")
print("BMI = 24")
print("Heart Rate = 166")
print("Age = 38")
print("Skin Resistance (0 for not dry, 1 for dry) = 0")

# Get input for each feature
coreTemp = get_input("Core Temp")
ambientTemp = get_input("Ambient Temp")
ambientHumidity = get_input("Ambient Humidity")
bmi = get_input("BMI")
heartRate = get_input("Heart Rate")
age = get_input("Age")
skinRes = get_input("Skin Resistance (0 for not dry, 1 for dry)")

# Prepare input data
input_data = {
    'coreTemp': tf.constant(coreTemp, dtype=tf.float32),
    'ambientTemp': tf.constant(ambientTemp, dtype=tf.float32),
    'ambientHumidity': tf.constant(ambientHumidity, dtype=tf.float32),
    'bmi': tf.constant(bmi, dtype=tf.float32),
    'heartRate': tf.constant(heartRate, dtype=tf.float32),
    'age': tf.constant(age, dtype=tf.float32),
    'skinRes': tf.constant(skinRes, dtype=tf.float32)
}

# Make predictions
output = loaded_model(input_data)

# Convert output to probability
probability = tf.sigmoid(output)

# Print the predicted probability
print("Predicted probability of having heatstroke:", probability.numpy())
