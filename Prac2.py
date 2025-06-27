import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset manually
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Prepare the data
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Highlight the more accurate predictions
highlight = abs(y_pred - y_test) < 5

# Plot the data and the model's predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test[highlight], y_pred[highlight], color='red', marker='o', label="Accurate")
plt.scatter(y_test, y_pred, color='blue', marker='x', label="Inaccurate")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.legend()
plt.show()

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-Squared:", r2)
