import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from os import path, makedirs
import random
from shutil import rmtree

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Get the banknote data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ["Variance", "Skewness", "Kurtosis", "Entropy", "Class"]
df = pd.read_csv(url, header=None, names=columns)

# Seperate data into input and output
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Reshape for ease
X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.reshape(1, -1), Y_test.reshape(1, -1)

# Create directories
DATA_DIR = "data"
MODELS_DIR = "models"
makedirs(DATA_DIR, exist_ok=True)
makedirs(MODELS_DIR, exist_ok=True)

# Empty models directory for cleaner structure
rmtree(MODELS_DIR)
makedirs(MODELS_DIR, exist_ok=True)

# Saving data that has to be used for testing later
np.save(path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(path.join(DATA_DIR, "Y_test.npy"), Y_test)

# I used 3 Layers, set value of Alpha as 0.001, used 100,000 epochs
EPOCHS = int(input("Number of epochs: "))
ALPHA = float(input("Value of Alpha: "))
LAYERS = int(input("Number of Layers: "))

# Number of features/neurons per layer. I used 4, 3 and 1 neuron in layer1, layer2 and layer3 respectively.
n0, m = X_train.shape
n_dict = {f"n{i}": int(input(f"Number of neurons in layer {i}: ")) for i in range(1, LAYERS+1)}
n_dict["n0"] = n0

# Asking about activation function
FUNCTION = input("Do you want to use Tanh in hidden layers or ReLU? ").lower()
if FUNCTION == "tanh":
    n_factor = 1
else:
    n_factor = 2

# Using dictionaries to regulate variables as for loops will be used
W_dict = {f"W{i}": np.random.rand(n_dict[f"n{i}"], n_dict[f"n{i-1}"]) * np.sqrt(n_factor / n_dict[f"n{i-1}"]) for i in range(1, LAYERS+1)}
b_dict = {f"b{i}": np.zeros((n_dict[f"n{i}"], 1)) for i in range(1, LAYERS+1)}
Z_dict = {}
A_dict = {"A0": X_train}
dZ_dict = {}
dW_dict = {}
db_dict = {}

for i in range(EPOCHS):
    # Forward Propagation
    for l in range(1, LAYERS+1):
        Z_dict[f"Z{l}"] = np.dot(W_dict[f"W{l}"], A_dict[f"A{l-1}"]) + b_dict[f"b{l}"]
        if l == LAYERS:
            A_dict[f"A{l}"] = 1 / (1 + np.exp(-Z_dict[f"Z{l}"]))
        else:
            if FUNCTION == "tanh":
                A_dict[f"A{l}"] = np.tanh(Z_dict[f"Z{l}"])
            else:
                A_dict[f"A{l}"] = np.maximum(0, Z_dict[f"Z{l}"])
    
    # Backward Propagation for the output layer
    dZ_dict[f"dZ{LAYERS}"] = A_dict[f"A{LAYERS}"] - Y_train
    dW_dict[f"dW{LAYERS}"] = 1/m * np.dot(dZ_dict[f"dZ{LAYERS}"], A_dict[f"A{LAYERS-1}"].T)
    db_dict[f"db{LAYERS}"] = 1/m * np.sum(dZ_dict[f"dZ{LAYERS}"], axis=1, keepdims=True)
    # Update gradients for output layer
    W_dict[f"W{LAYERS}"] -= ALPHA * dW_dict[f"dW{LAYERS}"]
    b_dict[f"b{LAYERS}"] -= ALPHA * db_dict[f"db{LAYERS}"]

    # Backward propagation for the rest of layers
    for l in range(LAYERS-1, 0, -1):
        if FUNCTION == "tanh":
            dZ_dict[f"dZ{l}"] = np.dot(W_dict[f"W{l+1}"].T, dZ_dict[f"dZ{l+1}"]) * (1 - np.square(A_dict[f"A{l}"]))
        else:
            dZ_dict[f"dZ{l}"] = np.dot(W_dict[f"W{l+1}"].T, dZ_dict[f"dZ{l+1}"]) * (Z_dict[f"Z{l}"] > 0).astype(int)
        dW_dict[f"dW{l}"] = 1/m * np.dot(dZ_dict[f"dZ{l}"], A_dict[f"A{l-1}"].T)
        db_dict[f"db{l}"] = 1/m * np.sum(dZ_dict[f"dZ{l}"], axis=1, keepdims=True)
        # Update gradients side by side
        W_dict[f"W{l}"] -= ALPHA * dW_dict[f"dW{l}"]
        b_dict[f"b{l}"] -= ALPHA * db_dict[f"db{l}"]
    
    # Calculate Loss
    Y_hat = A_dict[f"A{LAYERS}"]
    J = -1/m * np.sum(Y_train * np.log(Y_hat) + (1-Y_train)*np.log(1-Y_hat))

    # Keep getting updated on the progress
    if i % 1000 == 0:
        print(f"Epochs: {i}, Cost: {J}")

# Saving Loss to be used later in accuracy calculaion
np.save(path.join(MODELS_DIR, "J.npy"), J)

# Save weights
c = 1
for i in W_dict:
    print(f"Final W{c}: {W_dict[i]}")
    np.save(path.join(MODELS_DIR, f"W{c}.npy"), W_dict[i])
    c += 1

# Save bias
c = 1
for i in b_dict:
    print(f"Final b{c}: {b_dict[i]}")
    np.save(path.join(MODELS_DIR, f"b{c}.npy"), b_dict[i])
    c += 1