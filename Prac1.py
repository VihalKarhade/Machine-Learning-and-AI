import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the mean squared error loss function
def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).mean()

# Define the minimal dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the weights and biases for the hidden layer and the output layer
weights_hidden = np.random.rand(3, 3)
weights_output = np.random.rand(3, 1)
bias_hidden = np.zeros((1, 3))
bias_output = np.zeros((1, 1))

# Train the MLP for 1000 epochs
for epoch in range(1000):
    # Forward pass
    layer_hidden = np.dot(X, weights_hidden) + bias_hidden
    activation_hidden = sigmoid(layer_hidden)
    layer_output = np.dot(activation_hidden, weights_output) + bias_output
    activation_output = sigmoid(layer_output)
   
    # Compute the mean squared error
    loss = mean_squared_error(activation_output, y)
   
    # Backward pass
    error_output = activation_output - y
    derivative_output = sigmoid_derivative(activation_output)
    delta_output = error_output * derivative_output
    error_hidden = delta_output.dot(weights_output.T)
    derivative_hidden = sigmoid_derivative(activation_hidden)
    delta_hidden = error_hidden * derivative_hidden
   
    # Update the weights and biases
    weights_output -= activation_hidden.T.dot(delta_output) * 0.1
    bias_output -= np.sum(delta_output, axis=0, keepdims=True) * 0.1
    weights_hidden -= X.T.dot(delta_hidden) * 0.1
    bias_hidden -= np.sum(delta_hidden, axis=0, keepdims=True) * 0.1
   
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss}')

# Test the MLP with a new input
x_test = np.array([1, 0, 0])
layer_hidden = np.dot(x_test, weights_hidden) + bias_hidden
activation_hidden = sigmoid(layer_hidden)
layer_output = np.dot(activation_hidden, weights_output) + bias_output
activation_output = sigmoid(layer_output)
print(f'Predicted output: {activation_output}')
