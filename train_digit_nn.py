import numpy as np
from tensorflow.keras.datasets import mnist

# --- Load data ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

# --- Activation functions ---
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# --- Init weights ---
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

# --- Train ---
lr = 0.01
epochs = 10
batch_size = 64

for epoch in range(epochs):
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train_onehot = y_train_onehot[idx]

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train_onehot[i:i+batch_size]

        z1 = x_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = softmax(z2)

        dz2 = (a2 - y_batch) / batch_size
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)
        dW1 = x_batch.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Evaluate
    z1_test = x_test @ W1 + b1
    a1_test = relu(z1_test)
    z2_test = a1_test @ W2 + b2
    a2_test = softmax(z2_test)
    acc = (np.argmax(a2_test, axis=1) == y_test).mean()
    print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {acc:.4f}")

# --- Save weights ---
np.savez("trained_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("✅ Weights saved to trained_weights.npz")
