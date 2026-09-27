import numpy as np
import matplotlib.pyplot as plt

#Data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
Y = np.array([50, 55, 60, 68, 75, 82, 88, 95])


# Parameters
w = 0
b = 0

learning_rate = 0.01
iterations = 1000

def predict(X):
  return w * X + b

def loss(X, Y):
  predictions = predict(X)
  return np.mean((predictions - Y) ** 2)



for i in range(iterations):
  predictions = predict(X)

  error = predictions - Y

  dw = (2 / len(X)) * np.sum(X * error)

  db = (2 / len(X)) * np.sum(error)

  w = w - learning_rate * dw
  b = b - learning_rate * db


print("w =", w)
print("b =", b)
print("Loss =", loss(X, Y))


# Plot data
plt.scatter(X, Y, label="Actual Data")

# Prediction line
predictions = predict(X)


plt.plot(X, predictions, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

plt.title("Linear Regression with Gradient Descent")

plt.legend()
plt.grid()

plt.show()