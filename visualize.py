import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('heatstroke.csv')

# Split the dataset into features (X) and target variable (y)
X = data.drop('heatstroke', axis=1)
y = data['heatstroke']

# Load the saved model
tfdf_model = tfdf.keras.load_model("tfdf_model")

# Calculate the accuracy of the model
predictions = tfdf_model.predict(X)
accuracy = accuracy_score(y, predictions)
print("Accuracy:", accuracy)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["No Heatstroke", "Heatstroke"], filled=True)
plt.show()
