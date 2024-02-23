import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="heatguard.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
input_data_type = input_details[0]['dtype']

# Print input details
print("Input tensor shape:", input_shape)
print("Input data type:", input_data_type)

# Visualize input shape
plt.figure()
plt.bar(range(len(input_shape)), input_shape)
plt.xlabel('Dimension')
plt.ylabel('Size')
plt.title('Input Tensor Shape')
plt.show()
