from locale import normalize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

df_train = pd.read_csv('heatstroke.csv')

# print(df_train.head())

df_features = df_train.copy()
df_labels = df_features.pop('heatstroke')

df_features = np.array(df_features)
print(df_features)

normalize = layers.Normalization()

normalize.adapt(df_features)

df_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

df_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                 optimizer = tf.keras.optimizers.Adam())

df_model.fit(df_features, df_labels, epochs=10)

# Split the data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)

# Normalize the features
normalize = layers.Normalization()
normalize.adapt(features_train)

# Define the model
model = tf.keras.Sequential([
    normalize,
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Define early stopping to stop training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# Train the model
history = model.fit(features_train, labels_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(features_test, labels_test)

print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
print(f"error already")
# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model summary")
model.summary()

# Test case
test_input = np.array([[35.26355357,40.87841697,42.5,0.1,22.99028723,105.1037412,55,0],
                       [38.40248704,39.34884773,41.5,0.1,20.21187289,126.9243927,25,0],
                       [35.07745494,38.51249752,43.1,0.1,20.17576277,128.8443363,51,0],
                       [37.93597552,40.50572875,42.6,0.1,22.46539597,121.6937356,50,0],
                       [31.65905312,35.80249013,36.9297012,0.196760063,20.18281622,153.5502455,22.47479303,1],
                       [25.05841733,35.65184866,36.57479554,0.074533244,20.45848206,105.4043965,35.46297772,0],
                       [27.98865051,36.18020598,36.95988877,0.093441065,19.31389512,121.2912307,48.31197108,1]
                       ])

test_output = np.array([1,1,1, 1, 0, 0, 0])

predictions = model.predict(test_input)
predictions = [1 if p >= 0.5 else 0 for p in predictions]

print(f"Predictions: {predictions}")
print(f"Expected: {test_output.tolist()}")