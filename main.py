import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

# Load the dataset
data = pd.read_csv('heatstroke.csv')

# Split the dataset into features (X) and target variable (y)
X = data.drop('heatstroke', axis=1)
y = data['heatstroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Convert the trained model to a TensorFlow Decision Forests model
tfdf_model = tfdf.keras.get_native_model(clf)

# Export the TensorFlow Decision Forests model
tfdf_model.save("tfdf_model")

# Convert the TensorFlow SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("tfdf_model")
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open("tfdf_model.tflite", "wb") as f:
    f.write(tflite_model)
