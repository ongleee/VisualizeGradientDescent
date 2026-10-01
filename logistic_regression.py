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

def predict_class(X, threshold=0.5):
    probability = predict_probability(X)

    return (probability >= threshold).astype(int)



for i in range(iterations):

    predictions = predict_probability(X)

    error = predictions - Y

    dw = (1 / len(X)) * np.sum(X * error)

    db = (1 / len(X)) * np.sum(error)

    w = w - learning_rate * dw

    b = b - learning_rate * db

def confusion_matrix(Y, predictions):
    TP = np.sum((Y == 1) & (predictions == 1))
    TN = np.sum((Y == 0) & (predictions == 0))
    FP = np.sum((Y == 0) & (predictions == 1))
    FN = np.sum((Y == 1) & (predictions == 0))

    return TP, TN, FP, FN

def accuracy(Y, predictions):
    TP, TN, FP, FN = confusion_matrix(Y, predictions)

    return (TP + TN) / (TP + TN + FP + FN)

def precision(Y, predictions):
    TP, TN, FP, FN = confusion_matrix(Y, predictions)

    return TP / (TP + FP)

def recall(Y, predictions):
    TP, TN, FP, FN = confusion_matrix(Y, predictions)

    return TP / (TP + FN)

def f1_score(Y, predictions):
    p = precision(Y, predictions)
    r = recall(Y, predictions)

    return 2 * (p * r) / (p + r)



predictions_03 = predict_class(X, 0.3)
predictions_05 = predict_class(X, 0.5)
predictions_07 = predict_class(X, 0.7)

print("Threshold 0.3:", predictions_03)
print("Threshold 0.5:", predictions_05)
print("Threshold 0.7:", predictions_07)



predictions = predict_class(X,0.3)

print("Accuracy =", accuracy(Y, predictions))
print("Precision =", precision(Y, predictions))
print("Recall =", recall(Y, predictions))
print("F1 =", f1_score(Y, predictions))


TP, TN, FP, FN = confusion_matrix(Y, predictions)

print("TP =", TP)
print("TN =", TN)
print("FP =", FP)
print("FN =", FN)


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