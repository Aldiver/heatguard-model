import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df_train = pd.read_csv('heatstroke.csv')

# Separate features and labels
df_features = df_train.drop(columns=['heatstroke'])
df_labels = df_train['heatstroke']

# Normalize features
# Define the normalization layer with the input shape explicitly specified
normalize = layers.Normalization(input_shape=(8,))
normalize.adapt(np.array(df_features))

# Split the data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)

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

# Save the TensorFlow Lite model
with open('heatguard.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved as heatguard.tflite")
