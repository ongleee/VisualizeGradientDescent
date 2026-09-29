import numpy as np
import matplotlib.pyplot as plt

# Hour studued
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# 0 = Fail, 1 = Pass
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


w = 0
b = 0

learning_rate = 0.1
iterations = 1000

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def predict_probability(X):
  z = w * X + b
  return sigmoid(z)

def loss(X, Y):
  predictions = predict_probability(X)

  epsilon = 1e-15

  predictions = np.clip(
    predictions,
    epsilon,
    1 - epsilon
  )
  return -np.mean(
        Y * np.log(predictions)
        + (1 - Y) * np.log(1 - predictions)
  )

def predict_class(X):

    probability = predict_probability(X)

    return (probability >= 0.7).astype(int)


for i in range(iterations):

    predictions = predict_probability(X)

    error = predictions - Y

    dw = (1 / len(X)) * np.sum(X * error)

    db = (1 / len(X)) * np.sum(error)

    w = w - learning_rate * dw

    b = b - learning_rate * db


print("w =", w)
print("b =", b)
print("Loss =", loss(X, Y))
print(predict_class(X))

hours = np.array([4.5, 5.5, 10])

print("Probability:")
print(predict_probability(hours))

print("Prediction:")
print(predict_class(hours))

x_values = np.linspace(0, 10, 100)

probabilities = predict_probability(x_values)

plt.scatter(X, Y, label="Actual Data")

plt.plot(
    x_values,
    probabilities,
    label="Probability"
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Threshold"
)

plt.xlabel("Hours Studied")
plt.ylabel("Probability")

plt.title("Logistic Regression")

plt.legend()
plt.grid()

plt.show()