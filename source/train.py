import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


def activation_function_hidden(Z):
    return ( np.exp(Z) - np.exp(-Z) ) / ( np.exp(Z) + np.exp(-Z))


def derivative_activation_function_hidden(A):
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

W_dict = {
    "W1": np.random.randn(n1, n0),
    "W2": np.random.randn(n2, n1),
    "W3": np.random.randn(n3, n1)
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

epochs = 10_000

for i in range(epochs):
    
    for l in range(1, layers+1):
        W = W_dict[f"W{l}"]
        A = A_dict[f"A{l-1}"]
        b = b_dict[f"b{l}"]
        Z_dict[f"Z{l}"] = np.dot(W.T, A) + b
        A_dict[f"A{l-1}"] = activation_function_hidden(Z_dict[f"Z{l}"])
    
