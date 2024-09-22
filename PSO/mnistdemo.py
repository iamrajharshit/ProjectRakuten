# from nn import softmax
# from nn.model import Model
# from nn.layers import Layer
# from nn.losses import CrossEntropyLoss
# from nn.pipeline import DataLoader
# import numpy as np
# import pandas as pd

# def one_hot(y, depth=10):
#     y_1hot = np.zeros((y.shape[0], 10))
#     y_1hot[np.arange(y.shape[0]), y] = 1
#     return y_1hot

# df = pd.read_csv('./dataset/train.csv')
# all_data = df.values

# np.random.shuffle(all_data)

# split = int(0.8 * all_data.shape[0])
# x_train = all_data[:split, 1:]
# x_test = all_data[split:, 1:]
# y_train = all_data[:split, 0]
# y_test = all_data[split:, 0]

# y_train = one_hot(y_train.astype('int'))
# y_test = one_hot(y_test.astype('int'))

# def accuracy(y, y_hat):
#     y = np.argmax(y, axis=1)
#     y_hat = np.argmax(y_hat, axis=1)

#     return np.mean(y==y_hat)

# def relu(x):
#     return np.maximum(x, 0)

# model = Model()
# model.add_layer(Layer(784, 10, softmax))
# #model.add_layer(Layer(64, 64, relu))
# #model.add_layer(Layer(64, 10, softmax))

# model.compile(CrossEntropyLoss, DataLoader, accuracy,
#               batches_per_epoch=x_train.shape[0] // 32 + 1,
#               n_workers=50, c1=1., c2=2.)
# model.fit(x_train, y_train, 1)
# y_hat = model.predict(x_test)

# print('Accuracy on test:', accuracy(y_test, y_hat))

# from nn import softmax
# from nn.model import Model
# from nn.layers import Layer
# from nn.losses import CrossEntropyLoss
# from nn.pipeline import DataLoader
# import numpy as np
# import pandas as pd

# def one_hot(y, depth=10):
#     y_1hot = np.zeros((y.shape[0], 10))
#     y_1hot[np.arange(y.shape[0]), y] = 1
#     return y_1hot

# df = pd.read_csv('./dataset/train.csv')
# all_data = df.values

# np.random.shuffle(all_data)

# split = int(0.8 * all_data.shape[0])
# x_train = all_data[:split, 1:]
# x_test = all_data[split:, 1:]
# y_train = all_data[:split, 0]
# y_test = all_data[split:, 0]

# y_train = one_hot(y_train.astype('int'))
# y_test = one_hot(y_test.astype('int'))

# def accuracy(y, y_hat):
#     y = np.argmax(y, axis=1)
#     y_hat = np.argmax(y_hat, axis=1)
#     return np.mean(y == y_hat)

# def relu(x):
#     return np.maximum(x, 0)

# # Initialize the model and loss tracker
# model = Model()
# model.add_layer(Layer(784, 10, softmax))
# # model.add_layer(Layer(64, 64, relu))
# # model.add_layer(Layer(64, 10, softmax))

# # List to store fitness (loss) values
# fitness_values = []

# # Modify the training loop to store the loss after each epoch
# class CustomCrossEntropyLoss(CrossEntropyLoss):
#     def __call__(self, wts):
#         loss = super().__call__(wts)
#         fitness_values.append(loss)  # Save the fitness (loss) value
#         return loss

# # Use the custom loss function
# model.compile(CustomCrossEntropyLoss, DataLoader, accuracy,
#               batches_per_epoch=x_train.shape[0] // 32 + 1,
#               n_workers=50, c1=1., c2=2.)

# # Train the model and record the fitness values
# model.fit(x_train, y_train, 1)

# # Predict and compute the accuracy on the test set
# y_hat = model.predict(x_test)
# print('Accuracy on test:', accuracy(y_test, y_hat))

# # Save the fitness values (losses) to a CSV file
# fitness_df = pd.DataFrame(fitness_values, columns=["loss"])
# fitness_df.to_csv('fitness_values.csv', index=False)
# print("Fitness values saved to fitness_values.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn import softmax
from nn.model import Model
from nn.layers import Layer
from nn.losses import CrossEntropyLoss
from nn.pipeline import DataLoader

# Helper function for one-hot encoding
def one_hot(y, depth=10):
    y_1hot = np.zeros((y.shape[0], depth))
    y_1hot[np.arange(y.shape[0]), y] = 1
    return y_1hot

# Load dataset
df = pd.read_csv('./dataset/train.csv')
all_data = df.values
np.random.shuffle(all_data)

# Split data into training and testing sets
split = int(0.8 * all_data.shape[0])
x_train = all_data[:split, 1:]
x_test = all_data[split:, 1:]
y_train = all_data[:split, 0]
y_test = all_data[split:, 0]

# One-hot encode the labels
y_train = one_hot(y_train.astype('int'))
y_test = one_hot(y_test.astype('int'))

# Accuracy calculation
def accuracy(y, y_hat):
    y = np.argmax(y, axis=1)
    y_hat = np.argmax(y_hat, axis=1)
    return np.mean(y == y_hat)

# ReLU activation function
def relu(x):
    return np.maximum(x, 0)

# Build the model
model = Model()
model.add_layer(Layer(784, 10, softmax))  # Using softmax for final output

# Compile the model
model.compile(CrossEntropyLoss, DataLoader, accuracy,
              batches_per_epoch=x_train.shape[0] // 32 + 1,
              n_workers=50, c1=1., c2=2.)

# List to save fitness values (accuracy) after each epoch
fitness_values = []

# Custom training loop with accuracy recording
for epoch in range(1, 11):  # Training for 10 epochs
    model.fit(x_train, y_train, 1)
    y_hat_train = model.predict(x_train)
    acc_train = accuracy(y_train, y_hat_train)
    fitness_values.append(acc_train)
    print(f'Epoch {epoch}: Training accuracy = {acc_train:.4f}')

# Predicting on the test set
y_hat = model.predict(x_test)
test_acc = accuracy(y_test, y_hat)
print('Accuracy on test:', test_acc)

# Plotting fitness values (accuracy) over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(fitness_values) + 1), fitness_values, marker='o', label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plotting y_test vs y_hat (true vs predicted)
plt.figure(figsize=(8, 5))
plt.scatter(np.argmax(y_test, axis=1), np.argmax(y_hat, axis=1), alpha=0.5)
plt.title('True Labels vs Predicted Labels')
plt.xlabel('True Labels (y_test)')
plt.ylabel('Predicted Labels (y_hat)')
plt.grid(True)
plt.show()
