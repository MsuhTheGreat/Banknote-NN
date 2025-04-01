import numpy as np
from os.path import join

DATA_DIR = "data"
MODELS_DIR = "models"
LAYERS = int(input("Number of layers used during training: "))

# Load the necessary variables
X_test = np.load(join(DATA_DIR, "X_test.npy"))
Y_test = np.load(join(DATA_DIR, "Y_test.npy"))
J = np.load(join(MODELS_DIR, "J.npy"))

# Using dictionaries to regulate variables
W_dict = {}
b_dict = {}
Z_dict = {}
A_dict = {"A0": X_test}
for i in range(1, LAYERS+1):
    W_dict[f"W{i}"] = np.load(join(MODELS_DIR, f"W{i}.npy"))
    b_dict[f"b{i}"] = np.load(join(MODELS_DIR, f"b{i}.npy"))

# Forward Propagation
for l in range(1, LAYERS+1):
    Z_dict[f"Z{l}"] = np.dot(W_dict[f"W{l}"], A_dict[f"A{l-1}"]) + b_dict[f"b{l}"]
    if l == LAYERS:
        A_dict[f"A{l}"] = 1 / (1 + np.exp(-Z_dict[f"Z{l}"]))
    else:
        A_dict[f"A{l}"] = np.tanh(Z_dict[f"Z{l}"])
        # A_dict[f"A{l}"] = (np.exp(Z_dict[f"Z{l}"]) - np.exp(-Z_dict[f"Z{l}"])) / (np.exp(Z_dict[f"Z{l}"]) + np.exp(-Z_dict[f"Z{l}"]))

# Accuracy calculation using my own custom formula based on Loss
J_sigmoid = 1 / (1 + np.exp(-J))
my_accuracy = (1 - J_sigmoid) * 200
print(f"Accuracy According To My Custom Formula: {my_accuracy}")

# Accuracy according to traditional method
Y_hat = A_dict[f"A{LAYERS}"]
predictions = (Y_hat >= 0.5).astype(int)
traditional_accuracy = np.mean(predictions == Y_test) * 100
print(f"Accuracy According To Traditional Formula: {traditional_accuracy}")