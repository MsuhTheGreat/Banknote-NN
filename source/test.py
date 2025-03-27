import numpy as np
from os.path import join
from sklearn.metrics import f1_score

DATA_DIR = "data"
MODELS_DIR = "models"
LAYERS = 3

X_test = np.load(join(DATA_DIR, "X_test.npy"))
Y_test = np.load(join(DATA_DIR, "Y_test.npy"))
J = np.load(join(MODELS_DIR, "J.npy"))

W_dict = {}
b_dict = {}
Z_dict = {}
A_dict = {"A0": X_test}
for i in range(1, LAYERS+1):
    W_dict[f"W{i}"] = np.load(join(MODELS_DIR, f"W{i}.npy"))
    b_dict[f"b{i}"] = np.load(join(MODELS_DIR, f"b{i}.npy"))

for l in range(1, LAYERS+1):
    Z_dict[f"Z{l}"] = np.dot(W_dict[f"W{l}"], A_dict[f"A{l-1}"]) + b_dict[f"b{l}"]
    if l == LAYERS:
        A_dict[f"A{l}"] = 1 / (1 + np.exp(-Z_dict[f"Z{l}"]))
    else:
        A_dict[f"A{l}"] = (np.exp(Z_dict[f"Z{l}"]) - np.exp(-Z_dict[f"Z{l}"])) / (np.exp(Z_dict[f"Z{l}"]) + np.exp(-Z_dict[f"Z{l}"]))

J_sigmoid = 1 / (1 + np.exp(-J))
my_accuracy = (1 - J_sigmoid) * 200
print(f"Accuracy According To My Custom Formula: {my_accuracy}")

Y_hat = A_dict[f"A{LAYERS}"]
predictions = (Y_hat >= 0.5).astype(int)
traditional_accuracy = np.mean(predictions == Y_test) * 100
print(f"Accuracy According To Traditional Formula: {traditional_accuracy}")