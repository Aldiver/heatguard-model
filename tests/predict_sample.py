import numpy as np
import tensorflow as tf

# Define the test input
test_input = np.array([[35.07745494, 38.51249752, 0.1, 20.17576277, 128.8443363, 51, 0],
                       [37.93597552, 40.50572875, 0.1, 22.46539597, 121.6937356, 50, 0],
                       [31.65905312, 35.80249013, 0.196760063, 20.18281622, 153.5502455, 22, 0],
                       [25.05841733, 35.65184866, 0.074533244, 20.45848206, 105.4043965, 35, 0],
                       [27.98865051, 36.18020598, 0.093441065, 19.31389512, 121.2912307, 48, 0]])

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="heatguard.tflite")
interpreter.allocate_tensors()

# Define the actual values
actual_values = [1, 1, 0, 0, 0]

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Iterate through each test input
for i, input_data in enumerate(test_input):
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], [input_data.astype(np.float32)])

    # Perform the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Apply thresholding (0.5) to get the predicted class
    # predicted_class = 1 if output_data[0][0] > 0.5 else 0

    # Print the predicted class
    # print(f"Test Input {i+1}: Predicted Class = {predicted_class}")
    print(f"Test Input {i+1}: Output Probability = {output_data[0][0]}, Actual Value = {actual_values[i]}")
