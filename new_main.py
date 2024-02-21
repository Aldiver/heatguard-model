import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset into a Pandas dataframe
data = pd.read_csv("heatstroke.csv")

# Split the dataset into features (X) and labels (y)
X = data.drop(columns=["heatstroke"])
y = data["heatstroke"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Concatenate features and labels for training and testing sets
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Load TF-DF
import tensorflow_decision_forests as tfdf

# Convert the dataset into a TensorFlow dataset
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="heatstroke")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="heatstroke")

# Train a Random Forest model
model = tfdf.keras.RandomForestModel()
model.fit(train_ds)

# Summary of the model structure
model.summary()

# Evaluate the model
model.evaluate(test_ds)

# Export the model to a SavedModel
model.save("model")
