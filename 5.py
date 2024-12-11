# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv('lab1_dataset.csv')
df.head()

# %%
X = df.drop(columns='target')
y = df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape

# %%
print(X_train_scaled)

# %%
input_size = X_train_scaled.shape[1]
hidden_size = 15
output_size = 1
learning_rate = 0.1
epochs = 1200

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
  return (1 - np.tanh(z)**2)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward_propagation(X, y, Z1, A1, Z2, A2):
    m = X.shape[0]
    dZ2 = A2 - y.reshape(m, 1)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

y_train_reshaped  = y_train.values.reshape(-1, 1)
loss_values = []
epoch_values = []
m = X_train_scaled.shape[0]
for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_propagation(X_train_scaled)

    dW1, db1, dW2, db2 = backward_propagation(X_train_scaled, y_train_reshaped, Z1, A1, Z2, A2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 10 == 0:
        loss = -np.mean(y_train_reshaped*np.log(A2) + (1 - y_train_reshaped)*np.log(1 - A2))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        loss_values.append(loss)
        epoch_values.append(epoch)

print("Training complete!")



# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(epoch_values, loss_values)
plt.title('Convergence of Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2


y_pred_proba = predict(X_test_scaled)
print(y_pred_proba.reshape(-1))

y_pred = (y_pred_proba >= 0.2).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(" \nScores : \n")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



