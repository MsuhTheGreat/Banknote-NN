import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import os


DATA_DIR = "data"
MODELS_DIR = "models"


def activation_function_hidden(Z):
    return ( np.exp(Z) - np.exp(-Z) ) / ( np.exp(Z) + np.exp(-Z))


def derivative_activation_function_hidden(Z):
    A = activation_function_hidden(Z)
    return 1 - np.square(A)


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

X_train = X_train.reshape(28*28, -1)
X_test = X_test.reshape(28*28, -1)
Y_train = Y_train.reshape(10, -1)
Y_test = Y_test.reshape(10, -1)

layers = 3
n0, m = X_train.shape
n1 = 128
n2 = 64
n3 = 10

np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_DIR, "Y_test.npy"), Y_test)

W_dict = {
    "W1": np.random.randn(n1, n0) * np.sqrt(1 / n0),
    "W2": np.random.randn(n2, n1) * np.sqrt(1 / n1),
    "W3": np.random.randn(n3, n2) * np.sqrt(1 / n2)
}
b_dict = {
    "b1": np.zeros((n1, 1)),
    "b2": np.zeros((n2, 1)),
    "b3": np.zeros((n3, 1))
}
Z_dict = {}
A_dict = {
    "A0": X_train
}
dZ_dict = {}
dW_dict = {}
db_dict = {}

alpha = 0.001
epochs = 10_000

for i in range(epochs):
    
    for l in range(1, layers+1):
        W = W_dict[f"W{l}"]
        A = A_dict[f"A{l-1}"]
        b = b_dict[f"b{l}"]
        Z_dict[f"Z{l}"] = np.dot(W, A) + b
        if l == layers:
            A_dict[f"A{l}"] = np.exp(Z_dict[f"Z{l}"]) / np.sum(np.exp(Z_dict[f"Z{l}"]), axis=0, keepdims=True)
        else:
            A_dict[f"A{l}"] = activation_function_hidden(Z_dict[f"Z{l}"])
    
    dZ_dict[f"dZ{layers}"] = A_dict[f"A{layers}"] - Y_train
    dW_dict[f"dW{layers}"] = 1/m * np.dot(dZ_dict[f"dZ{layers}"], A_dict[f"A{layers-1}"].T)
    db_dict[f"db{layers}"] = 1/m * np.sum(dZ_dict[f"dZ{layers}"], axis=1, keepdims=True)

    for l in range(layers-1, 0, -1):
        W = W_dict[f"W{l+1}"].T
        dZ = dZ_dict[f"dZ{l+1}"]
        A = A_dict[f"A{l-1}"]
        Z = Z_dict[f"Z{l}"]
        dZ_dict[f"dZ{l}"] = np.dot(W, dZ) * derivative_activation_function_hidden(Z)
        dW_dict[f"dW{l}"] = 1/m * np.dot(dZ_dict[f"dZ{l}"], A.T)
        db_dict[f"db{l}"] = 1/m * np.sum(dZ_dict[f"dZ{l}"], axis=1, keepdims=True)
    
    for l in range(1, layers+1):
        W_dict[f"W{l}"] = W_dict[f"W{l}"] - alpha * dW_dict[f"dW{l}"]
        b_dict[f"b{l}"] = b_dict[f"b{l}"] - alpha * db_dict[f"db{l}"]
    
    J = -1/m * np.sum(Y_train * np.log(A_dict[f"A{layers}"]))

    if i % 100 == 0:
        print(f"Epochs: {i}, Cost: {J}")
    
print("Final Weights:-")
c = 1
for i in W_dict:
    print(f"\tW{c}: {W_dict[i]}")
    np.save(os.path.join(MODELS_DIR, f"W{c}.npy"), f"W{c}")
    c += 1

print("Final Bias:-")
c = 1
for i in b_dict:
    print(f"\tb{c}: {b_dict[i]}")
    np.save(os.path.join(MODELS_DIR, f"b{c}.npy"), f"b{c}")
    c += 1
