
import pickle
import os
import pandas as pd
import numpy as np
import time # To time our training

print("Imports successful!")

# data loading

train_file = "extended_mnist_train.pkl"
test_file = "extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

# data training

train_data_list = []
train_labels_list = []
for image, label in train:
    train_data_list.append(image.flatten())
    train_labels_list.append(label)

test_data_list = []

for image, image_id in test:
    test_data_list.append(image.flatten())

X_train = np.array(train_data_list, dtype=np.float32)
y_train = np.array(train_labels_list)

X_test = np.array(test_data_list, dtype=np.float32)

print(f"Original X_train shape: {X_train.shape}")  # (m, 784)
print(f"Original y_train shape: {y_train.shape}")  # (m,)
print(f"Original X_test shape: {X_test.shape}")  # (k, 784)

    # normalizare
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Max pixel value in X_train: {X_train.max()}")

# One-Hot Encode Labels

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

NUM_CLASSES = 10
y_train_one_hot = one_hot_encode(y_train, NUM_CLASSES)

print(f"Original label: {y_train[0]}")
print(f"One-hot label: {y_train_one_hot[0]}")
print(f"Shape of one-hot labels: {y_train_one_hot.shape}") # (m, 10)


# NN Class

class NumPyPerceptronNetwork:

    def __init__(self, input_size, output_size):
        """
        Initialize the model's parameters (Weights and Biases).
        input_size = 784
        output_size = 10
        """
        # We initialize weights with small random numbers.
        # This breaks symmetry and allows the network to learn.
        # Shape: (784, 10)
        self.W = np.random.randn(input_size, output_size) * 0.01

        # We can initialize biases to zero.
        # Shape: (1, 10) (The 1 makes broadcasting easier)
        self.b = np.zeros((1, output_size))

        print("Model initialized.")
        print(f"Weights shape: {self.W.shape}")
        print(f"Biases shape: {self.b.shape}")

    def softmax(self, z):
        """
        The Softmax activation function.
        It converts raw scores (logits) into probabilities.
        """
        # 'z' has shape (batch_size, 10)

        # This is a trick for "numeric stability".
        # Subtracting the max value from z prevents 'np.exp' from
        # creating infinitely large numbers, which would crash the program.
        # The final result is mathematically identical.
        stable_z = z - np.max(z, axis=1, keepdims=True)

        # Calculate exponents
        exp_z = np.exp(stable_z)

        # Divide each exponent by the sum of all exponents in its row
        # This makes each row sum up to 1 (a probability distribution)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Performs the forward propagation step.
        X shape: (batch_size, 784)
        """
        # 1. Linear combination (the "raw vote count")
        # Z = X * W + b
        # (batch_size, 784) @ (784, 10) -> (batch_size, 10)
        # We add 'b' (1, 10), and NumPy "broadcasts" it to
        # add to every row in the (batch_size, 10) matrix.
        z = np.dot(X, self.W) + self.b  # calculating logits

        # 2. Activation function (the "percentages")
        # 'a' (activations) will also have shape (batch_size, 10)
        a = self.softmax(z)

        return a

    def compute_loss(self, y_true_one_hot, y_pred_proba):
        """
        Computes the Cross-Entropy Loss (the "penalty").
        y_true_one_hot shape: (batch_size, 10)
        y_pred_proba shape: (batch_size, 10)
        """
        m = y_true_one_hot.shape[0]  # batch_size

        # This is the formula for cross-entropy loss
        # L = - (1/m) * SUM( y_true * log(y_pred) )

        # We add a tiny value (1e-9) to np.log to prevent
        # log(0) which is -infinity and would crash the program.
        log_likelihood = -np.log(y_pred_proba[np.arange(m), np.argmax(y_true_one_hot, axis=1)] + 1e-9)
        loss = np.sum(log_likelihood) / m

        # A simpler (but equivalent) way using the one-hot matrix:
        # loss = - (1 / m) * np.sum(y_true_one_hot * np.log(y_pred_proba + 1e-9))

        return loss

    def backward(self, X, y_true_one_hot, y_pred_proba):
        """
        Performs the backward propagation step (the "blame game").
        Calculates the gradients (the "blame") for W and b.

        X shape: (batch_size, 784)
        y_true_one_hot shape: (batch_size, 10)
        y_pred_proba shape: (batch_size, 10)
        """
        m = X.shape[0]  # batch_size

        # 1. Gradient of the Loss w.r.t. Z (the "raw vote counts")
        # This has a very simple and clean derivative: (pred_proba - true_label)
        # dL/dZ = A - Y
        dz = y_pred_proba - y_true_one_hot
        # We average it over the batch
        dz = dz / m

        # 2. Gradient of the Loss w.r.t. W (the "blame for weights")
        # dL/dW = X_transpose * dL/dZ
        # (784, batch_size) @ (batch_size, 10) -> (784, 10)
        # This matches the shape of self.W. Perfect!
        dw = np.dot(X.T, dz)

        # 3. Gradient of the Loss w.r.t. b (the "blame for biases")
        # dL/db = sum(dL/dZ) over the batch
        # We sum along axis=0 to squish all rows into one.
        # (batch_size, 10) -> (1, 10)
        # This matches the shape of self.b. Perfect!
        db = np.sum(dz, axis=0, keepdims=True)

        return dw, db

    def update_params(self, dw, db, learning_rate):
        """
        Performs the parameter update (the "correction").
        """
        # Simple gradient descent
        self.W = self.W - (learning_rate * dw)
        self.b = self.b - (learning_rate * db)

    def train(self, X_train, y_train_one_hot, learning_rate, epochs, batch_size):
        """
        The main training loop.
        Puts all the pieces together.
        """
        num_examples = X_train.shape[0]

        print(f"Starting training...")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")

        for epoch in range(epochs):
            start_time = time.time()

            # --- Mini-Batch Logic ---
            # Shuffle the data at the start of each epoch
            # This is important for the model to learn robustly
            permutation = np.random.permutation(num_examples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train_one_hot[permutation]

            epoch_loss = 0.0

            for i in range(0, num_examples, batch_size):
                # Get the current mini-batch
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]

                # 1. Forward Pass
                y_pred = self.forward(X_batch)

                # 2. Compute Loss
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss

                # 3. Backward Pass
                dw, db = self.backward(X_batch, y_batch, y_pred)

                # 4. Update Parameters
                self.update_params(dw, db, learning_rate)

            # --- End of Epoch ---
            avg_loss = epoch_loss / (num_examples / batch_size)
            end_time = time.time()

            # Print a progress report
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Time: {end_time - start_time:.2f}s")

    def predict(self, X):
        """
        Makes predictions on new, unseen data (like X_test).
        """
        # Run the forward pass to get probabilities
        probabilities = self.forward(X)

        # Find the index (0-9) with the highest probability
        # for each example (row) in the batch.
        # np.argmax with axis=1 does exactly this.
        predictions = np.argmax(probabilities, axis=1)

        return predictions